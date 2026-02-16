import os

# Check composio
f1 = r'd:\composio_apps.json'
if os.path.exists(f1):
    size = os.path.getsize(f1)
    print(f"composio_apps.json: {size} bytes")
    if size > 0:
        with open(f1, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()[:3000]
            print(content)
    else:
        print("  (empty file)")
else:
    print("  (file not found)")

print("\n" + "="*60)

# Check autogpt plugins
f2 = r'd:\autogpt_plugins.md'
if os.path.exists(f2):
    size = os.path.getsize(f2)
    print(f"autogpt_plugins.md: {size} bytes")
    if size > 0:
        with open(f2, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()[:3000]
            print(content)
    else:
        print("  (empty file)")
else:
    print("  (file not found)")

print("\n" + "="*60)

# Check Composio public API
import urllib.request
import json

apis_to_check = [
    ("Composio Apps API", "https://api.composio.dev/api/v1/apps"),
    ("Composio Actions API", "https://api.composio.dev/api/v1/actions?limit=5"),
    ("Toolhouse API", "https://api.toolhouse.ai/v1/tools"),
    ("LangChain Hub", "https://api.hub.langchain.com/repos?limit=5"),
]

for name, url in apis_to_check:
    print(f"\nChecking: {name}")
    print(f"  URL: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = resp.read().decode('utf-8', errors='replace')[:1500]
        print(f"  Status: {resp.status}")
        print(f"  Response: {data[:500]}")
    except Exception as e:
        print(f"  Error: {e}")
