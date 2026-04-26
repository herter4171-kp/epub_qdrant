#!/usr/bin/env python3
"""Dump full catalog from MCP server to JSON for outline generation."""
import json
import requests

MCP_URL = "http://localhost:8090/mcp"
COLLECTIONS = ["books-semantic", "papers-semantic"]

def list_sources(collection, limit=200, offset=0):
    resp = requests.post(MCP_URL, json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "list_sources",
            "arguments": {"collection": collection, "limit": limit, "offset": offset}
        }
    })
    data = json.loads(resp.json()["result"]["content"][0]["text"])
    return data

catalog = {}
for coll in COLLECTIONS:
    print(f"Fetching {coll}...")
    first = list_sources(coll, limit=0)
    total = first["total"]
    categories = first.get("categories", {})
    print(f"  {total} sources, categories: {categories}")
    
    all_sources = []
    offset = 0
    while offset < total:
        batch = list_sources(coll, limit=50, offset=offset)
        all_sources.extend(batch["sources"])
        offset += 50
        print(f"  fetched {len(all_sources)}/{total}")
    
    catalog[coll] = {
        "total": total,
        "categories": categories,
        "sources": all_sources
    }

with open("catalog_dump.json", "w") as f:
    json.dump(catalog, f, indent=2)

print(f"\nDone. Wrote catalog_dump.json")
print(f"  books-semantic: {catalog['books-semantic']['total']} sources")
print(f"  papers-semantic: {catalog['papers-semantic']['total']} sources")
