#!/usr/bin/env python3
"""Count nonzero activations from sparse vectors in Qdrant collection."""

import json
import numpy as np
import matplotlib.pyplot as plt
import requests
import argparse
import os
from collections import Counter


def scroll_all_points(host: str, collection: str, output_dir: str):
    """Scroll all points from a Qdrant collection and save sparse vector stats."""
    
    base_url = f"http://{host}"
    points = []
    next_page_offset = None
    
    print(f"Connecting to Qdrant at {base_url}")
    print(f"Collection: {collection}")
    
    # First, get collection info
    resp = requests.get(f"{base_url}/collections/{collection}")
    if resp.status_code != 200:
        print(f"Error: Collection {collection} not found")
        return
    
    collection_info = resp.json()
    points_count = collection_info["result"]["points_count"]
    print(f"Total points: {points_count}")
    
    # Scroll all points
    batch_size = 1000
    point_count = 0
    
    while True:
        url = f"{base_url}/collections/{collection}/points/scroll"
        payload = {
            "limit": batch_size,
            "with_vector": True
        }
        if next_page_offset is not None:
            payload["offset"] = next_page_offset
        
        resp = requests.post(url, json=payload)
        data = resp.json()
        
        if data["status"] != "ok":
            print(f"Error: {data}")
            break
        
        batch_points = data["result"]["points"]
        if not batch_points:
            break
        
        points.extend(batch_points)
        point_count += len(batch_points)
        
        next_page_offset = data["result"].get("next_page_offset")
        if next_page_offset is None:
            break
        
        print(f"Scrolled {point_count}/{points_count} points", end="\r")
    
    print(f"\nTotal points collected: {len(points)}")
    
    # Count nonzero values in each sparse vector
    nonzero_counts = []
    for point in points:
        if "vector" in point and "sparse" in point["vector"]:
            values = point["vector"]["sparse"]["values"]
            nonzero_count = sum(1 for v in values if v > 0)
            nonzero_counts.append(nonzero_count)
    
    print(f"Analyzed {len(nonzero_counts)} sparse vectors")
    
    # Compute statistics
    nonzero_counts = np.array(nonzero_counts)
    mean_nz = np.mean(nonzero_counts)
    median_nz = np.median(nonzero_counts)
    p10 = np.percentile(nonzero_counts, 10)
    p90 = np.percentile(nonzero_counts, 90)
    min_nz = np.min(nonzero_counts)
    max_nz = np.max(nonzero_counts)
    
    print(f"\nNonzero activation statistics:")
    print(f"  Mean:   {mean_nz:.2f}")
    print(f"  Median: {median_nz:.2f}")
    print(f"  P10:    {p10:.2f}")
    print(f"  P90:    {p90:.2f}")
    print(f"  Min:    {min_nz}")
    print(f"  Max:    {max_nz}")
    
    # Save stats to file
    stats_file = os.path.join(output_dir, "nonzero_stats.json")
    stats = {
        "collection": collection,
        "points_count": len(points),
        "nonzero_counts": {
            "mean": float(mean_nz),
            "median": float(median_nz),
            "p10": float(p10),
            "p90": float(p90),
            "min": int(min_nz),
            "max": int(max_nz)
        }
    }
    
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_file}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(nonzero_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Nonzero Activation Count')
    plt.ylabel('Frequency')
    plt.title(f'SPLADE Nonzero Activation Distribution\n'
              f'Collection: {collection} | Points: {len(points)}')
    plt.axvline(mean_nz, color='red', linestyle='--', 
                label=f'Mean: {mean_nz:.2f}')
    plt.axvline(median_nz, color='green', linestyle='--', 
                label=f'Median: {median_nz:.2f}')
    plt.axvline(p10, color='orange', linestyle='--', 
                label=f'P10: {p10:.2f}')
    plt.axvline(p90, color='purple', linestyle='--', 
                label=f'P90: {p90:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = os.path.join(output_dir, "nonzero_distribution.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    
    # Count distribution of point counts
    nz_counter = Counter(nonzero_counts)
    print(f"\nValue distribution (top 10):")
    for val, count in sorted(nz_counter.items(), key=lambda x: -x[1])[:10]:
        pct = count / len(nonzero_counts) * 100
        print(f"  {val}: {count} ({pct:.1f}%)")
    
    # Save K value for config
    print(f"\n=== K VALUE (for config.yaml): {int(round(mean_nz))} ===")
    
    # Also save the actual nonzero_counts array for potential future use
    counts_file = os.path.join(output_dir, "nonzero_counts.npy")
    np.save(counts_file, nonzero_counts)
    print(f"Counts array saved to {counts_file}")


def main():
    parser = argparse.ArgumentParser(description='Count nonzero activations from sparse vectors')
    parser.add_argument('--host', default='192.168.68.75:6333',
                        help='Qdrant host:port')
    parser.add_argument('--collection', default='sparse-only-256len',
                        help='Qdrant collection name')
    parser.add_argument('--output-dir', default='./scripts/sae_impl',
                        help='Output directory for stats and plots')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    scroll_all_points(args.host, args.collection, args.output_dir)


if __name__ == "__main__":
    main()