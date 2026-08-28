import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
#!/usr/bin/env python
import argparse, json, pathlib
from market_maker.evaluation.backtest import compare_all
def main():
    parser=argparse.ArgumentParser(description="Backtest")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", type=str, default="reports/backtest.json")
    args=parser.parse_args()
    res=compare_all(env_steps=args.steps)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output,"w") as f:
        json.dump(res,f,indent=2)
    print(json.dumps(res,indent=2))
if __name__=="__main__":
    main()

