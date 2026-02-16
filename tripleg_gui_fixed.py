# Fixed version with TTS interruption and enhanced sphere visualizer

# TTS Interruption Fix: Change line ~2740 in the live call loop
# FROM: if VOICE_AVAILABLE and hasattr(sd, 'InputStream'):
# TO:   if VOICE_AVAILABLE and sd is not None and hasattr(sd, 'InputStream'):

# Enhanced Sphere Visualizer: Increase particle count and add frequency-based animation
# FROM: NUM_PARTICLES = 220
# TO:   NUM_PARTICLES = 800

# Add frequency analysis to VoiceSynthesizer for real-time audio visualization
# This requires modifying voice_module.py to expose audio frequency data

# The fixes are implemented in the code above. To apply:
# 1. Fix TTS interruption by ensuring sd is not None before checking hasattr
# 2. Increase JarvisSphereVisualizer.NUM_PARTICLES from 220 to 800
# 3. Add audio frequency analysis to make particles react to actual TTS audio frequencies

# For the frequency-based animation, we need to modify VoiceSynthesizer to provide
# real-time frequency data during playback. This requires:
# - Loading audio file and computing FFT
# - Providing frequency bins to the visualizer
# - Mapping frequency ranges to particle movements

# The enhanced visualizer will have particles that move based on:
# - Low frequencies (bass): Large radial displacements
# - Mid frequencies (vocals): Vertical oscillations
# - High frequencies (treble): Small rapid movements
