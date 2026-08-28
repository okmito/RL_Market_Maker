from __future__ import annotations
import numpy as np

def sharpe(returns: list[float], risk_free: float=0.0)->float:
    if len(returns)<2:
        return 0.0
    arr=np.array(returns)
    std=np.std(arr, ddof=1)
    if std==0:
        return 0.0
    return float((np.mean(arr)-risk_free)/std*np.sqrt(252*24*60))  # approx per minute

def max_drawdown(equity: list[float])->float:
    if not equity:
        return 0.0
    peak=equity[0]
    max_dd=0.0
    for v in equity:
        if v>peak:
            peak=v
        dd=(peak-v)/peak if peak else 0
        if dd>max_dd:
            max_dd=dd
    return float(max_dd)

def compute_metrics(portfolio_history: list[float], inventory_history: list[float], fills: list[dict], fees: float)->dict:
    returns=np.diff(portfolio_history)/np.array(portfolio_history[:-1]) if len(portfolio_history)>1 else np.array([0])
    total_pnl = portfolio_history[-1]-portfolio_history[0] if portfolio_history else 0
    return {
        "total_pnl": float(total_pnl),
        "sharpe": sharpe(returns.tolist()),
        "max_drawdown": max_drawdown(portfolio_history),
        "inventory_mean": float(np.mean(np.abs(inventory_history))) if inventory_history else 0,
        "inventory_var": float(np.var(inventory_history)) if inventory_history else 0,
        "fill_count": len(fills),
        "fees": float(fees),
        "final_portfolio": float(portfolio_history[-1]) if portfolio_history else 0,
    }
