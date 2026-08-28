from __future__ import annotations
import time
import hmac
import hashlib
import aiohttp
from decimal import Decimal
from typing import Optional
from urllib.parse import urlencode

from loguru import logger

from market_maker.exchange.base import ExchangeBase, Order, OrderSide, OrderStatus, Balance, SymbolInfo

class BinanceTestnetExchange(ExchangeBase):
    """Binance Testnet adapter (spot). Uses testnet.binance.vision endpoints. Production endpoints are blocked."""
    def __init__(self, api_key: Optional[str]=None, api_secret: Optional[str]=None, base_url: str="https://testnet.binance.vision", symbol: str="BTCUSDT"):
        if not base_url.startswith("https://testnet.binance.vision"):
            raise ValueError("Only testnet base_url allowed. Production trading is disabled.")
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol
        self._session: Optional[aiohttp.ClientSession]=None

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        logger.info(f"BinanceTestnet connected {self.base_url}")

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time()*1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        sig = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    async def _request(self, method: str, path: str, params: dict = None, signed: bool=False):
        if self._session is None:
            await self.connect()
        params = params or {}
        headers = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("API credentials missing for signed request")
            params = self._sign(params)
            headers["X-MBX-APIKEY"] = self.api_key
        url = f"{self.base_url}{path}"
        async with self._session.request(method, url, params=params, headers=headers) as resp:
            data = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"Binance error {resp.status}: {data}")
            return data

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        data = await self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol})
        s = data["symbols"][0]
        tick = Decimal("0.01")
        step = Decimal("0.00001")
        min_notional = Decimal("10")
        for f in s["filters"]:
            if f["filterType"]=="PRICE_FILTER":
                tick = Decimal(f["tickSize"])
            if f["filterType"]=="LOT_SIZE":
                step = Decimal(f["stepSize"])
            if f["filterType"] in ("MIN_NOTIONAL","NOTIONAL"):
                min_notional = Decimal(f.get("minNotional", f.get("minNotional", "10")))
        return SymbolInfo(symbol=symbol, base_asset=s["baseAsset"], quote_asset=s["quoteAsset"],
                          price_tick=tick, qty_step=step, min_qty=Decimal(s["filters"][1]["minQty"]) if len(s["filters"])>1 else Decimal("0.00001"),
                          max_qty=Decimal("1000"), min_notional=min_notional)

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self._request("GET", "/api/v3/depth", {"symbol": symbol, "limit": limit})

    async def get_mid_price(self, symbol: str) -> Optional[Decimal]:
        data = await self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol})
        bid = Decimal(data["bidPrice"])
        ask = Decimal(data["askPrice"])
        return (bid+ask)/2

    async def place_order(self, order: Order) -> Order:
        params = {"symbol": order.symbol, "side": order.side.value, "type": order.type.value,
                  "quantity": str(order.quantity)}
        if order.price:
            params["price"] = str(order.price)
            params["timeInForce"] = order.time_in_force.value
        if order.client_order_id:
            params["newClientOrderId"] = order.client_order_id
        data = await self._request("POST", "/api/v3/order", params, signed=True)
        order.order_id = data["orderId"]
        order.status = OrderStatus[data["status"]]
        order.executed_qty = Decimal(data.get("executedQty","0"))
        return order

    async def cancel_order(self, symbol: str, order_id: int) -> Order:
        data = await self._request("DELETE", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)
        o = Order(symbol=symbol, side=OrderSide[data["side"]], type=OrderType[data["type"]], quantity=Decimal(data["origQty"]), price=Decimal(data["price"]) if data.get("price") else None, order_id=data["orderId"], status=OrderStatus[data["status"]])
        return o

    async def cancel_all_orders(self, symbol: str) -> list[Order]:
        data = await self._request("DELETE", "/api/v3/openOrders", {"symbol": symbol}, signed=True)
        return []

    async def get_open_orders(self, symbol: str) -> list[Order]:
        data = await self._request("GET", "/api/v3/openOrders", {"symbol": symbol}, signed=True)
        res=[]
        for d in data:
            res.append(Order(symbol=d["symbol"], side=OrderSide[d["side"]], type=d["type"], quantity=Decimal(d["origQty"]), price=Decimal(d["price"]) if d.get("price") else None, order_id=d["orderId"], status=OrderStatus[d["status"]], executed_qty=Decimal(d["executedQty"])))
        return res

    async def get_order(self, symbol: str, order_id: int) -> Optional[Order]:
        data = await self._request("GET", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)
        return Order(symbol=data["symbol"], side=OrderSide[data["side"]], type=data["type"], quantity=Decimal(data["origQty"]), price=Decimal(data["price"]) if data.get("price") else None, order_id=data["orderId"], status=OrderStatus[data["status"]])

    async def get_balances(self) -> dict[str, Balance]:
        data = await self._request("GET", "/api/v3/account", {}, signed=True)
        out={}
        for b in data["balances"]:
            out[b["asset"]]=Balance(asset=b["asset"], free=Decimal(b["free"]), locked=Decimal(b["locked"]))
        return out
