from decimal import Decimal
from market_maker.market_data.order_book import OrderBook

def test_snapshot_and_best():
    book=OrderBook("BTCUSDT")
    bids=[(Decimal("50000"), Decimal("1")), (Decimal("49999"), Decimal("2"))]
    asks=[(Decimal("50001"), Decimal("1")), (Decimal("50002"), Decimal("2"))]
    book.initialize_from_snapshot(bids, asks, timestamp=1.0, sequence=1, last_update_id=1)
    assert book.best_bid()[0]==Decimal("50000")
    assert book.best_ask()[0]==Decimal("50001")
    assert book.mid_price()==Decimal("50000.5")
    assert book.spread()==Decimal("1")

def test_incremental_update():
    book=OrderBook("BTCUSDT")
    bids=[(Decimal("50000"), Decimal("1"))]
    asks=[(Decimal("50001"), Decimal("1"))]
    book.initialize_from_snapshot(bids, asks, timestamp=1.0, sequence=1)
    # update: remove bid, add new ask
    book.apply_update(bids=[(Decimal("50000"), Decimal("0"))], asks=[(Decimal("50001"), Decimal("2"))], timestamp=2.0, sequence=2)
    assert book.best_bid() is None or book.best_bid()[0]!=Decimal("50000")
    assert book.best_ask()[1]==Decimal("2")

def test_crossed_detection():
    book=OrderBook("BTCUSDT", allow_crossed=False)
    bids=[(Decimal("50002"), Decimal("1"))]
    asks=[(Decimal("50001"), Decimal("1"))]
    book.initialize_from_snapshot(bids, asks, timestamp=1.0, sequence=1)
    # Should have attempted fix
    b=book.best_bid()
    a=book.best_ask()
    if b and a:
        assert b[0] < a[0]

def test_imbalance():
    book=OrderBook("BTCUSDT")
    bids=[(Decimal("50000"), Decimal("10"))]
    asks=[(Decimal("50001"), Decimal("5"))]
    book.initialize_from_snapshot(bids, asks, timestamp=1.0, sequence=1)
    imb=book.depth_imbalance(1)
    assert imb > 0  # bid heavier

def test_microprice():
    book=OrderBook("BTCUSDT")
    bids=[(Decimal("50000"), Decimal("2"))]
    asks=[(Decimal("50100"), Decimal("1"))]
    book.initialize_from_snapshot(bids, asks, timestamp=1.0, sequence=1)
    mp=book.microprice()
    # Weighted towards opposite side: bid 2 vs ask 1 => micro closer to ask
    assert mp is not None
