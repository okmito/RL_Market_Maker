import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
#!/usr/bin/env python
"""Generate synthetic LOB data for training."""
import argparse, pathlib, random, time, json
from decimal import Decimal

def main():
    parser=argparse.ArgumentParser(description="Generate synthetic LOB data")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--output", type=str, default="data/synthetic_lob.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args=parser.parse_args()
    random.seed(args.seed)
    price=50000.0
    path=pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f:
        for i in range(args.steps):
            price*= (1+ random.gauss(0,0.001))
            spread=random.uniform(5,20)
            bid=price*(1-spread/10000/2)
            ask=price*(1+spread/10000/2)
            bids=[[bid - j*0.5, random.uniform(0.5,5)] for j in range(5)]
            asks=[[ask + j*0.5, random.uniform(0.5,5)] for j in range(5)]
            rec={"timestamp": time.time()+i*0.1, "sequence":i, "mid":price, "bids":bids, "asks":asks}
            f.write(json.dumps(rec)+"\n")
    print(f"Generated {args.steps} snapshots to {path}")

if __name__=="__main__":
    main()

