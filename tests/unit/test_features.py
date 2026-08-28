from decimal import Decimal
from market_maker.market_data.order_book import OrderBook
from market_maker.features.lob_features import LOBFeatures, MicrostructureFeatures

def test_lob_features():
    book=OrderBook("BTCUSDT")
    bids=[(Decimal(str(50000-i)), Decimal("1")) for i in range(10)]
    asks=[(Decimal(str(50001+i)), Decimal("1")) for i in range(10)]
    book.initialize_from_snapshot(bids, asks, timestamp=1.0, sequence=1)
    lf=LOBFeatures(depth_levels=5)
    feats=lf.extract(book)
    assert "mid_price" in feats
    assert "spread_bps" in feats

def test_microstructure():
    mf=MicrostructureFeatures(window=10)
    mf.add_trade(50000, 0.1, "buy", 1.0)
    mf.add_trade(50001, 0.2, "sell", 2.0)
    feats=mf.compute()
    assert "order_flow_imbalance" in feats
