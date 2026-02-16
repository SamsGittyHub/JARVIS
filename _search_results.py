import json

for f in [r'd:\search3.json', r'd:\search4.json', r'd:\search5.json']:
    print(f"=== {f} ===")
    try:
        d = json.load(open(f))
        print(f"Total: {d.get('total_count', 0)}")
        for r in d.get('items', []):
            desc = str(r.get('description', ''))[:80]
            print(f"  {r['full_name']} ({r['stargazers_count']}*) - {desc}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
