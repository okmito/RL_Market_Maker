import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
#!/usr/bin/env python
import os, asyncio, sys
from dotenv import load_dotenv
load_dotenv()
async def main():
    key=os.getenv("BINANCE_TESTNET_API_KEY")
    secret=os.getenv("BINANCE_TESTNET_API_SECRET")
    if not key or not secret:
        print("HUMAN ACTION REQUIRED: BINANCE_TESTNET_API_KEY / SECRET not set. Testnet tests will be skipped.")
        print("Set them in .env to enable testnet connectivity test.")
        sys.exit(0)
    from market_maker.exchange.binance_testnet import BinanceTestnetExchange
    ex=BinanceTestnetExchange(api_key=key, api_secret=secret)
    try:
        await ex.connect()
        info=await ex.get_symbol_info("BTCUSDT")
        print(f"Symbol info: {info}")
        book=await ex.get_order_book("BTCUSDT", limit=5)
        print(f"Order book top: {book}")
        mid=await ex.get_mid_price("BTCUSDT")
        print(f"Mid price: {mid}")
        balances=await ex.get_balances()
        print(f"Balances: {list(balances.keys())[:3]}")
        print("Testnet connectivity VERIFIED")
    except Exception as e:
        print(f"Testnet connectivity failed: {e}")
    finally:
        await ex.disconnect()
if __name__=="__main__":
    asyncio.run(main())

