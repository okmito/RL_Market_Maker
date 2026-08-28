import os, pytest
from dotenv import load_dotenv
load_dotenv()

@pytest.mark.skipif(not os.getenv("BINANCE_TESTNET_API_KEY"), reason="No testnet creds")
@pytest.mark.asyncio
async def test_testnet():
    from market_maker.exchange.binance_testnet import BinanceTestnetExchange
    ex=BinanceTestnetExchange(api_key=os.getenv("BINANCE_TESTNET_API_KEY"), api_secret=os.getenv("BINANCE_TESTNET_API_SECRET"))
    await ex.connect()
    info=await ex.get_symbol_info("BTCUSDT")
    assert info.symbol=="BTCUSDT"
    await ex.disconnect()
