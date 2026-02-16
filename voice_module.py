"""
TripleG Voice Module
====================
Provides voice input (Whisper STT) and voice output (Edge-TTS) for the TripleG AI agent.

Components:
- VoiceRecorder: Captures audio from microphone
- WhisperTranscriber: Transcribes audio to text using local OpenAI Whisper (large model)
- VoiceSynthesizer: Converts text to speech using Edge-TTS
- VoiceManager: Orchestrates the full voice pipeline

Dependencies:
    pip install openai-whisper sounddevice numpy edge-tts pygame

Note: First run will download the Whisper large model (~1.5GB)
"""

import os
import sys
import time
import tempfile
import threading
import queue
import asyncio
from pathlib import Path
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

# ==========================================
# DEPENDENCY CHECKS
# ==========================================

SOUNDDEVICE_AVAILABLE = False
WHISPER_AVAILABLE = False
EDGE_TTS_AVAILABLE = False
PYGAME_AVAILABLE = False
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        _gpu_name = torch.cuda.get_device_name(0)
        _vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[VoiceModule] ✓ CUDA available: {_gpu_name} ({_vram:.1f}GB VRAM)")
    else:
        print("[VoiceModule] ⚠ CUDA NOT available — Whisper will be VERY slow on CPU!")
except ImportError:
    print("[VoiceModule] WARNING: torch not installed. CUDA acceleration unavailable.")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    print("[VoiceModule] WARNING: sounddevice not installed. Run: pip install sounddevice")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("[VoiceModule] WARNING: openai-whisper not installed. Run: pip install openai-whisper")

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    print("[VoiceModule] WARNING: edge-tts not installed. Run: pip install edge-tts")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    print("[VoiceModule] WARNING: pygame not installed. Run: pip install pygame")


# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class VoiceConfig:
    """Voice module configuration."""
    # Whisper settings
    whisper_model: str = "large"  # tiny, base, small, medium, large, large-v2, large-v3
    whisper_language: Optional[str] = "en"  # Force English for speed (skip language detection)
    whisper_device: str = "cuda"  # "cuda" ONLY — CPU is too slow for large model
    
    # Whisper speed optimizations (CUDA)
    whisper_beam_size: int = 1  # 1 = greedy (fastest), 5 = beam search (slower but better)
    whisper_best_of: int = 1  # Number of candidates (1 = fastest)
    whisper_fp16: bool = True  # Use FP16 on CUDA (2x faster, less VRAM)
    whisper_condition_on_previous_text: bool = False  # Disable for speed (avoids hallucination loops)
    whisper_no_speech_threshold: float = 0.6  # Skip silent segments faster
    whisper_compression_ratio_threshold: float = 2.4  # Filter out garbage transcriptions
    
    # Recording settings
    sample_rate: int = 16000  # Whisper expects 16kHz
    channels: int = 1  # Mono
    dtype: str = "float32"
    silence_threshold: float = 0.01  # RMS threshold for silence detection
    silence_duration: float = 1.5  # Seconds of silence to stop recording
    max_recording_duration: float = 60.0  # Maximum recording length in seconds
    
    # TTS settings — JARVIS-style (British, clear, efficient)
    tts_voice: str = "en-GB-RyanNeural"  # British male — closest to JARVIS
    tts_rate: str = "+15%"  # Faster delivery for responsiveness
    tts_volume: str = "+0%"  # Volume adjustment
    tts_pitch: str = "+0Hz"  # Neutral pitch (faster processing)
    
    # Cache directory
    cache_dir: Path = Path.home() / ".tripleg" / "voice_cache"


class VoiceState(Enum):
    """Voice module states."""
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    SPEAKING = "speaking"
    ERROR = "error"


# ==========================================
# VOICE RECORDER
# ==========================================

class VoiceRecorder:
    """
    Records audio from the microphone.
    Supports both manual stop and automatic silence detection.
    Also supports continuous VAD (Voice Activity Detection) for hands-free mode.
    """
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.is_recording = False
        self.audio_data: list = []
        self._stream = None
        self._silence_start: Optional[float] = None
        
        # VAD (Voice Activity Detection) for continuous listening
        self._vad_enabled = False
        self._vad_listening = False
        self._vad_speech_detected = False
        self._vad_speech_start: Optional[float] = None
        self._vad_callback: Optional[Callable[[np.ndarray], None]] = None
        self._vad_status_callback: Optional[Callable[[str, str], None]] = None
        self._vad_min_speech_duration = 0.3  # Minimum speech duration to capture (seconds)
        self._vad_pre_speech_buffer: list = []  # Buffer to capture audio just before speech
        self._vad_pre_speech_duration = 0.3  # Seconds of audio to keep before speech detected
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"[VoiceRecorder] Stream status: {status}")
        if self.is_recording:
            self.audio_data.append(indata.copy())
            
            # Check for silence (for auto-stop feature)
            rms = np.sqrt(np.mean(indata**2))
            if rms < self.config.silence_threshold:
                if self._silence_start is None:
                    self._silence_start = time.time()
                elif time.time() - self._silence_start > self.config.silence_duration:
                    self.is_recording = False  # Auto-stop on silence
            else:
                self._silence_start = None
    
    def start_recording(self) -> bool:
        """Start recording from microphone."""
        if not SOUNDDEVICE_AVAILABLE:
            print("[VoiceRecorder] ERROR: sounddevice not available")
            return False
            
        try:
            self.audio_data = []
            self.is_recording = True
            self._silence_start = None
            
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                callback=self._audio_callback
            )
            self._stream.start()
            print("[VoiceRecorder] Recording started...")
            return True
        except Exception as e:
            print(f"[VoiceRecorder] ERROR starting recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and return audio data as numpy array."""
        self.is_recording = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        if not self.audio_data:
            print("[VoiceRecorder] No audio data recorded")
            return None
        
        # Concatenate all audio chunks
        audio = np.concatenate(self.audio_data, axis=0)
        audio = audio.flatten()  # Ensure 1D array
        
        print(f"[VoiceRecorder] Recording stopped. Duration: {len(audio) / self.config.sample_rate:.2f}s")
        return audio
    
    def record_until_silence(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Record until silence is detected or timeout.
        Blocking call.
        """
        if not self.start_recording():
            return None
        
        timeout = timeout or self.config.max_recording_duration
        start_time = time.time()
        
        while self.is_recording:
            if time.time() - start_time > timeout:
                print("[VoiceRecorder] Recording timeout reached")
                break
            time.sleep(0.1)
        
        return self.stop_recording()
    
    def get_input_devices(self) -> list:
        """List available input devices."""
        if not SOUNDDEVICE_AVAILABLE:
            return []
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    input_devices.append({
                        'index': i,
                        'name': d['name'],
                        'channels': d['max_input_channels'],
                        'sample_rate': d['default_samplerate']
                    })
            return input_devices
        except Exception as e:
            print(f"[VoiceRecorder] ERROR listing devices: {e}")
            return []


# ==========================================
# WHISPER TRANSCRIBER
# ==========================================

class WhisperTranscriber:
    """
    Transcribes audio to text using OpenAI Whisper (local model).
    Uses the large model on CUDA for best accuracy + speed.
    REFUSES to run on CPU — too slow for large model.
    """
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.model = None
        self._model_loaded = False
        self._loading = False
        self._device_used = "unknown"
        
    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Load the Whisper model on CUDA. Downloads if not cached (~1.5GB for large).
        Will NOT fall back to CPU — raises error instead.
        """
        if not WHISPER_AVAILABLE:
            print("[WhisperTranscriber] ERROR: whisper not available")
            return False
            
        if self._model_loaded:
            return True
            
        if self._loading:
            print("[WhisperTranscriber] Model already loading...")
            return False
        
        self._loading = True
        model_name = self.config.whisper_model
        device = self.config.whisper_device
        
        try:
            # ── CUDA enforcement ──
            if device == "cuda" and not CUDA_AVAILABLE:
                msg = (
                    "[WhisperTranscriber] FATAL: CUDA requested but NOT available!\n"
                    "  The 'large' Whisper model is too slow on CPU.\n"
                    "  Install CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
                    "  Or use a smaller model (tiny/base) for CPU."
                )
                print(msg)
                if progress_callback:
                    progress_callback("ERROR: CUDA not available! Install CUDA PyTorch.")
                self._loading = False
                return False
            
            if progress_callback:
                progress_callback(f"Loading Whisper {model_name} on {device.upper()}...")
            
            print(f"[WhisperTranscriber] Loading Whisper {model_name} model...")
            print(f"[WhisperTranscriber] Device: {device.upper()} (ENFORCED — no CPU fallback)")
            if CUDA_AVAILABLE:
                gpu_name = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                print(f"[WhisperTranscriber] GPU: {gpu_name} | VRAM: {vram_free:.1f}GB free / {vram_total:.1f}GB total")
            
            print(f"[WhisperTranscriber] First run downloads ~1.5GB model file...")
            
            # Load model on CUDA
            self.model = whisper.load_model(
                model_name,
                device=device
            )
            self._device_used = device
            
            # Verify model is on CUDA
            if device == "cuda" and TORCH_AVAILABLE:
                param_device = str(next(self.model.parameters()).device)
                if "cuda" not in param_device:
                    print(f"[WhisperTranscriber] WARNING: Model loaded on {param_device}, expected cuda!")
                else:
                    print(f"[WhisperTranscriber] ✓ Model confirmed on {param_device}")
            
            self._model_loaded = True
            self._loading = False
            
            if progress_callback:
                progress_callback(f"✓ Whisper {model_name} loaded on {device.upper()}!")
            
            print(f"[WhisperTranscriber] ✓ Model loaded successfully on {device.upper()}!")
            return True
            
        except Exception as e:
            self._loading = False
            print(f"[WhisperTranscriber] ERROR loading model: {e}")
            
            # DO NOT fall back to CPU — it's too slow for large model
            if self.config.whisper_device == "cuda":
                print("[WhisperTranscriber] ✗ CUDA load failed. NOT falling back to CPU (too slow for large model).")
                print("[WhisperTranscriber]   Fix CUDA installation or use whisper_model='base' for CPU.")
                if progress_callback:
                    progress_callback(f"ERROR: CUDA load failed: {e}")
            
            return False
    
    def transcribe(self, audio: np.ndarray, 
                   progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, dict]:
        """
        Transcribe audio to text with CUDA-optimized parameters.
        
        Args:
            audio: Audio data as numpy array (float32, 16kHz)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (transcribed_text, metadata_dict)
        """
        if not self._model_loaded:
            if not self.load_model(progress_callback):
                return "", {"error": "Model not loaded"}
        
        try:
            if progress_callback:
                progress_callback("Transcribing audio (CUDA)...")
            
            print(f"[WhisperTranscriber] Transcribing on {self._device_used.upper()}...")
            start_time = time.time()
            
            # Ensure audio is float32 and normalized
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0  # Normalize if int16 range
            
            # ── CUDA-optimized transcription parameters ──
            use_fp16 = self.config.whisper_fp16 and self._device_used == "cuda"
            
            result = self.model.transcribe(
                audio,
                language=self.config.whisper_language,  # "en" = skip language detection (faster)
                fp16=use_fp16,  # FP16 on CUDA = 2x faster, less VRAM
                beam_size=self.config.whisper_beam_size,  # 1 = greedy decoding (fastest)
                best_of=self.config.whisper_best_of,  # 1 = no reranking (fastest)
                condition_on_previous_text=self.config.whisper_condition_on_previous_text,  # False = faster, no hallucination loops
                no_speech_threshold=self.config.whisper_no_speech_threshold,  # Skip silent segments
                compression_ratio_threshold=self.config.whisper_compression_ratio_threshold,  # Filter garbage
                verbose=False,
            )
            
            elapsed = time.time() - start_time
            text = result.get("text", "").strip()
            audio_duration = len(audio) / self.config.sample_rate
            rtf = elapsed / audio_duration if audio_duration > 0 else 0  # Real-time factor
            
            metadata = {
                "language": result.get("language", "unknown"),
                "duration": audio_duration,
                "transcription_time": elapsed,
                "real_time_factor": rtf,
                "segments": len(result.get("segments", [])),
                "device": self._device_used,
                "fp16": use_fp16,
            }
            
            speed_indicator = "🚀" if rtf < 0.5 else "⚡" if rtf < 1.0 else "🐌"
            print(f"[WhisperTranscriber] {speed_indicator} Transcribed in {elapsed:.2f}s (RTF: {rtf:.2f}x) on {self._device_used.upper()}: '{text[:80]}'")
            
            if progress_callback:
                progress_callback(f"Transcribed in {elapsed:.1f}s: {text[:50]}...")
            
            return text, metadata
            
        except Exception as e:
            print(f"[WhisperTranscriber] ERROR transcribing: {e}")
            return "", {"error": str(e)}
    
    def transcribe_file(self, audio_path: str) -> Tuple[str, dict]:
        """Transcribe audio from a file with CUDA optimizations."""
        if not self._model_loaded:
            if not self.load_model():
                return "", {"error": "Model not loaded"}
        
        try:
            use_fp16 = self.config.whisper_fp16 and self._device_used == "cuda"
            result = self.model.transcribe(
                audio_path,
                language=self.config.whisper_language,
                fp16=use_fp16,
                beam_size=self.config.whisper_beam_size,
                best_of=self.config.whisper_best_of,
                condition_on_previous_text=self.config.whisper_condition_on_previous_text,
                verbose=False,
            )
            return result.get("text", "").strip(), {
                "language": result.get("language", "unknown"),
                "segments": len(result.get("segments", [])),
                "device": self._device_used,
            }
        except Exception as e:
            return "", {"error": str(e)}
    
    @property
    def is_loaded(self) -> bool:
        return self._model_loaded
    
    @property
    def device(self) -> str:
        return self._device_used


# ==========================================
# VOICE SYNTHESIZER (TTS)
# ==========================================

class VoiceSynthesizer:
    """
    Text-to-Speech using Microsoft Edge TTS.
    Free, high-quality voices, no API key required.
    """
    
    # Popular Edge TTS voices
    VOICES = {
        # English
        "en-US-GuyNeural": "English (US) - Guy (Male)",
        "en-US-JennyNeural": "English (US) - Jenny (Female)",
        "en-US-AriaNeural": "English (US) - Aria (Female)",
        "en-US-DavisNeural": "English (US) - Davis (Male)",
        "en-GB-RyanNeural": "English (UK) - Ryan (Male)",
        "en-GB-SoniaNeural": "English (UK) - Sonia (Female)",
        # Other languages
        "es-ES-AlvaroNeural": "Spanish (Spain) - Alvaro (Male)",
        "fr-FR-HenriNeural": "French - Henri (Male)",
        "de-DE-ConradNeural": "German - Conrad (Male)",
        "ja-JP-KeitaNeural": "Japanese - Keita (Male)",
        "zh-CN-YunxiNeural": "Chinese - Yunxi (Male)",
    }
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pygame_initialized = False
        
    def _init_pygame(self):
        """Initialize pygame mixer for audio playback."""
        if not PYGAME_AVAILABLE:
            return False
        if self._pygame_initialized:
            return True
        try:
            pygame.mixer.init()
            self._pygame_initialized = True
            return True
        except Exception as e:
            print(f"[VoiceSynthesizer] ERROR initializing pygame: {e}")
            return False
    
    async def _synthesize_async(self, text: str, output_path: str) -> bool:
        """Async synthesis using edge-tts with JARVIS-like voice tuning."""
        try:
            # JARVIS characteristics: British male, slightly lower pitch, calm measured pace
            pitch = getattr(self.config, 'tts_pitch', '-5Hz')
            
            communicate = edge_tts.Communicate(
                text,
                self.config.tts_voice,
                rate=self.config.tts_rate,
                volume=self.config.tts_volume,
                pitch=pitch,
            )
            await communicate.save(output_path)
            return True
        except Exception as e:
            print(f"[VoiceSynthesizer] ERROR in async synthesis: {e}")
            return False
    
    def _run_async(self, coro):
        """Run an async coroutine efficiently, reusing event loop when possible."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # Already in an async context — use a new thread with its own loop
            result = [None]
            def _run():
                result[0] = asyncio.run(coro)
            t = threading.Thread(target=_run)
            t.start()
            t.join()
            return result[0]
        else:
            # No running loop — safe to use asyncio.run()
            return asyncio.run(coro)
    
    def synthesize(self, text: str, 
                   output_path: Optional[str] = None,
                   progress_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """
        Synthesize text to speech and save to file.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path. If None, uses temp file.
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the generated audio file, or None on error
        """
        if not EDGE_TTS_AVAILABLE:
            print("[VoiceSynthesizer] ERROR: edge-tts not available")
            return None
        
        if not text.strip():
            return None
        
        try:
            if progress_callback:
                progress_callback("Synthesizing speech...")
            
            # Generate output path if not provided
            if output_path is None:
                output_path = str(self.config.cache_dir / f"tts_{int(time.time())}.mp3")
            
            print(f"[VoiceSynthesizer] Synthesizing: '{text[:50]}...'")
            start = time.time()
            
            # Run async synthesis (with efficient loop reuse)
            self._run_async(self._synthesize_async(text, output_path))
            
            elapsed = time.time() - start
            
            if os.path.exists(output_path):
                size_kb = os.path.getsize(output_path) / 1024
                print(f"[VoiceSynthesizer] ✓ Synthesized in {elapsed:.2f}s ({size_kb:.0f}KB): {output_path}")
                return output_path
            else:
                print("[VoiceSynthesizer] ERROR: Output file not created")
                return None
                
        except Exception as e:
            print(f"[VoiceSynthesizer] ERROR synthesizing: {e}")
            return None
    
    def speak(self, text: str, 
              blocking: bool = True,
              progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Synthesize and play text as speech.
        
        Args:
            text: Text to speak
            blocking: If True, wait for playback to complete
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        if not self._init_pygame():
            print("[VoiceSynthesizer] Cannot play audio: pygame not available")
            return False
        
        # Synthesize to temp file
        audio_path = self.synthesize(text, progress_callback=progress_callback)
        if not audio_path:
            return False
        
        try:
            if progress_callback:
                progress_callback("Playing audio...")
            
            print("[VoiceSynthesizer] Playing audio...")
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            self._current_audio_path = audio_path
            
            if blocking:
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                print("[VoiceSynthesizer] Playback complete")
            
            return True
            
        except Exception as e:
            print(f"[VoiceSynthesizer] ERROR playing audio: {e}")
            return False
        finally:
            # Clean up temp file after playback
            if blocking:
                try:
                    os.remove(audio_path)
                except:
                    pass
    
    def speak_nonblocking(self, text: str,
                          progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Synthesize and start playing text as speech (non-blocking).
        Use is_playing() to check status, stop() to interrupt.
        
        Returns:
            True if playback started successfully
        """
        if not self._init_pygame():
            print("[VoiceSynthesizer] Cannot play audio: pygame not available")
            return False
        
        audio_path = self.synthesize(text, progress_callback=progress_callback)
        if not audio_path:
            return False
        
        try:
            if progress_callback:
                progress_callback("Playing audio...")
            
            print("[VoiceSynthesizer] Playing audio (non-blocking)...")
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            self._current_audio_path = audio_path
            return True
            
        except Exception as e:
            print(f"[VoiceSynthesizer] ERROR playing audio: {e}")
            return False
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if self._pygame_initialized:
            try:
                return pygame.mixer.music.get_busy()
            except:
                return False
        return False
    
    def stop(self):
        """Stop any currently playing audio."""
        if self._pygame_initialized:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        # Clean up audio file
        if hasattr(self, '_current_audio_path') and self._current_audio_path:
            try:
                os.remove(self._current_audio_path)
            except:
                pass
            self._current_audio_path = None
    
    def set_voice(self, voice: str):
        """Change the TTS voice."""
        self.config.tts_voice = voice
        print(f"[VoiceSynthesizer] Voice set to: {voice}")
    
    @classmethod
    def list_voices(cls) -> dict:
        """Return available voices."""
        return cls.VOICES.copy()


# ==========================================
# VOICE MANAGER
# ==========================================

class VoiceManager:
    """
    High-level voice manager that orchestrates:
    - Recording from microphone
    - Transcription via Whisper
    - Text-to-speech playback
    
    Provides callbacks for GUI integration.
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.recorder = VoiceRecorder(self.config)
        self.transcriber = WhisperTranscriber(self.config)
        self.synthesizer = VoiceSynthesizer(self.config)
        
        self.state = VoiceState.IDLE
        self._state_callback: Optional[Callable[[VoiceState, str], None]] = None
        self._transcription_callback: Optional[Callable[[str], None]] = None
        
        # Background thread for non-blocking operations
        self._worker_thread: Optional[threading.Thread] = None
        self._task_queue: queue.Queue = queue.Queue()
        self._running = False
    
    def set_state_callback(self, callback: Callable[[VoiceState, str], None]):
        """Set callback for state changes. Callback receives (state, message)."""
        self._state_callback = callback
    
    def set_transcription_callback(self, callback: Callable[[str], None]):
        """Set callback for transcription results. Callback receives transcribed text."""
        self._transcription_callback = callback
    
    def _update_state(self, state: VoiceState, message: str = ""):
        """Update state and notify callback."""
        self.state = state
        print(f"[VoiceManager] State: {state.value} - {message}")
        if self._state_callback:
            self._state_callback(state, message)
    
    def initialize(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Initialize the voice system (load Whisper model on CUDA).
        This can take a while on first run (model download).
        """
        self._update_state(VoiceState.IDLE, "Initializing...")
        
        # Check dependencies
        missing = []
        if not SOUNDDEVICE_AVAILABLE:
            missing.append("sounddevice")
        if not WHISPER_AVAILABLE:
            missing.append("openai-whisper")
        if not EDGE_TTS_AVAILABLE:
            missing.append("edge-tts")
        if not PYGAME_AVAILABLE:
            missing.append("pygame")
        
        if missing:
            msg = f"Missing dependencies: {', '.join(missing)}"
            self._update_state(VoiceState.ERROR, msg)
            return False
        
        # CUDA check
        if self.config.whisper_device == "cuda" and not CUDA_AVAILABLE:
            msg = "CUDA not available! Install CUDA PyTorch for fast Whisper."
            self._update_state(VoiceState.ERROR, msg)
            if progress_callback:
                progress_callback(f"ERROR: {msg}")
            return False
        
        if progress_callback:
            progress_callback(f"Loading Whisper {self.config.whisper_model} on CUDA...")
        
        # Load Whisper model on CUDA
        if not self.transcriber.load_model(progress_callback):
            self._update_state(VoiceState.ERROR, "Failed to load Whisper model")
            return False
        
        device_info = f"CUDA ({self.transcriber.device})" if self.transcriber.device == "cuda" else self.transcriber.device
        self._update_state(VoiceState.IDLE, f"Ready — Whisper on {device_info}")
        return True
    
    def start_recording(self) -> bool:
        """Start recording from microphone."""
        if self.state != VoiceState.IDLE:
            print(f"[VoiceManager] Cannot start recording in state: {self.state}")
            return False
        
        if self.recorder.start_recording():
            self._update_state(VoiceState.RECORDING, "Recording...")
            return True
        else:
            self._update_state(VoiceState.ERROR, "Failed to start recording")
            return False
    
    def stop_recording_and_transcribe(self) -> Optional[str]:
        """
        Stop recording and transcribe the audio.
        Blocking call - returns transcribed text.
        """
        if self.state != VoiceState.RECORDING:
            print(f"[VoiceManager] Not recording, state: {self.state}")
            return None
        
        # Stop recording
        audio = self.recorder.stop_recording()
        if audio is None or len(audio) < self.config.sample_rate * 0.5:  # Less than 0.5s
            self._update_state(VoiceState.IDLE, "Recording too short")
            return None
        
        # Transcribe
        self._update_state(VoiceState.TRANSCRIBING, "Transcribing...")
        text, metadata = self.transcriber.transcribe(audio)
        
        if text:
            self._update_state(VoiceState.IDLE, f"Transcribed: {text[:30]}...")
            if self._transcription_callback:
                self._transcription_callback(text)
            return text
        else:
            self._update_state(VoiceState.IDLE, "No speech detected")
            return None
    
    def stop_recording_and_transcribe_async(self):
        """
        Stop recording and transcribe in background thread.
        Result delivered via transcription_callback.
        """
        if self.state != VoiceState.RECORDING:
            return
        
        def _worker():
            self.stop_recording_and_transcribe()
        
        threading.Thread(target=_worker, daemon=True).start()
    
    def speak(self, text: str, blocking: bool = False):
        """
        Speak text using TTS.
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        if not text.strip():
            return
        
        def _speak():
            self._update_state(VoiceState.SPEAKING, "Speaking...")
            self.synthesizer.speak(text, blocking=True)
            self._update_state(VoiceState.IDLE, "Ready")
        
        if blocking:
            _speak()
        else:
            threading.Thread(target=_speak, daemon=True).start()
    
    def stop_speaking(self):
        """Stop any currently playing speech."""
        self.synthesizer.stop()
        if self.state == VoiceState.SPEAKING:
            self._update_state(VoiceState.IDLE, "Stopped")
    
    def cancel(self):
        """Cancel any ongoing operation."""
        if self.state == VoiceState.RECORDING:
            self.recorder.stop_recording()
        elif self.state == VoiceState.SPEAKING:
            self.synthesizer.stop()
        self._update_state(VoiceState.IDLE, "Cancelled")
    
    def record_and_transcribe(self, 
                               timeout: Optional[float] = None,
                               progress_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """
        Full pipeline: Record until silence, then transcribe.
        Blocking call.
        
        Args:
            timeout: Maximum recording duration
            progress_callback: Optional progress callback
            
        Returns:
            Transcribed text or None
        """
        self._update_state(VoiceState.RECORDING, "Recording (speak now)...")
        
        if progress_callback:
            progress_callback("Recording... (speak now)")
        
        audio = self.recorder.record_until_silence(timeout)
        
        if audio is None or len(audio) < self.config.sample_rate * 0.5:
            self._update_state(VoiceState.IDLE, "No audio captured")
            return None
        
        self._update_state(VoiceState.TRANSCRIBING, "Transcribing...")
        
        if progress_callback:
            progress_callback("Transcribing...")
        
        text, metadata = self.transcriber.transcribe(audio, progress_callback)
        
        self._update_state(VoiceState.IDLE, "Ready")
        
        return text if text else None
    
    @property
    def is_ready(self) -> bool:
        """Check if voice system is ready."""
        return self.state == VoiceState.IDLE and self.transcriber.is_loaded
    
    @property
    def is_recording(self) -> bool:
        return self.state == VoiceState.RECORDING
    
    @property
    def is_speaking(self) -> bool:
        return self.state == VoiceState.SPEAKING


# ==========================================
# CONVENIENCE FUNCTIONS
# ==========================================

def check_dependencies() -> dict:
    """Check which voice dependencies are available."""
    return {
        "sounddevice": SOUNDDEVICE_AVAILABLE,
        "whisper": WHISPER_AVAILABLE,
        "edge_tts": EDGE_TTS_AVAILABLE,
        "pygame": PYGAME_AVAILABLE,
        "all_available": all([
            SOUNDDEVICE_AVAILABLE,
            WHISPER_AVAILABLE,
            EDGE_TTS_AVAILABLE,
            PYGAME_AVAILABLE
        ])
    }


def get_install_command() -> str:
    """Get pip install command for missing dependencies."""
    missing = []
    if not SOUNDDEVICE_AVAILABLE:
        missing.append("sounddevice")
    if not WHISPER_AVAILABLE:
        missing.append("openai-whisper")
    if not EDGE_TTS_AVAILABLE:
        missing.append("edge-tts")
    if not PYGAME_AVAILABLE:
        missing.append("pygame")
    
    if missing:
        return f"pip install {' '.join(missing)}"
    return "All dependencies installed!"


# ==========================================
# TEST / DEMO
# ==========================================

def demo():
    """Demo the voice module."""
    print("=" * 60)
    print("TripleG Voice Module Demo")
    print("=" * 60)
    
    # Check dependencies
    deps = check_dependencies()
    print(f"\nDependencies: {deps}")
    
    if not deps["all_available"]:
        print(f"\nMissing dependencies! Run:\n  {get_install_command()}")
        return
    
    # Create voice manager
    config = VoiceConfig(whisper_model="large")
    manager = VoiceManager(config)
    
    # Initialize (loads Whisper model)
    print("\nInitializing voice system...")
    if not manager.initialize():
        print("Failed to initialize!")
        return
    
    print("\n" + "=" * 60)
    print("Voice system ready!")
    print("Press Enter to start recording, then speak.")
    print("Recording will stop after 1.5s of silence.")
    print("=" * 60)
    
    input("\nPress Enter to start recording...")
    
    # Record and transcribe
    text = manager.record_and_transcribe(timeout=30)
    
    if text:
        print(f"\n✓ Transcribed: {text}")
        
        # Speak it back
        print("\nSpeaking the transcription...")
        manager.speak(f"You said: {text}", blocking=True)
        print("Done!")
    else:
        print("\n✗ No speech detected")


if __name__ == "__main__":
    demo()
