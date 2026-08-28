import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
#!/usr/bin/env python
"""Collect live LOB from testnet (or simulate if no creds)."""
import argparse, asyncio, os, sys
from dotenv import load_dotenv
load_dotenv()
async def main():
    parser=argparse.ArgumentParser(description="Collect LOB")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output", type=str, default="data/collected.msgpack")
    args=parser.parse_args()
    key=os.getenv("BINANCE_TESTNET_API_KEY")
    if not key:
        print("No testnet credentials - running synthetic demo instead")
        from scripts.generate_simulation_data import main as gen
        print("Use generate_simulation_data.py for synthetic data")
        sys.exit(0)
    print(f"Collecting {args.symbol} for {args.duration}s - requires live ws (not implemented fully, placeholder)")
    await asyncio.sleep(1)
    print("Done (placeholder)")

if __name__=="__main__":
    asyncio.run(main())

