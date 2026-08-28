from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import time
import random

app=FastAPI(title="RL Market Maker", version="0.1.0")

@app.get("/")
def root():
    return {"status":"ok", "service":"rl-market-maker", "docs":"/docs", "dashboard":"/dashboard"}

@app.get("/health")
def health():
    return {"status":"healthy", "timestamp": time.time()}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>RL Market Maker — Dashboard</title>
<style>
:root{--bg:#0b0e14;--card:#151a27;--muted:#9aa3b2;--accent:#6ee7b7;--accent2:#60a5fa;--danger:#f87171;--warn:#fbbf24;--border:#222a3a;--text:#e6eaf2}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial;background:var(--bg);color:var(--text)}
a{color:var(--accent2);text-decoration:none}
.header{position:sticky;top:0;background:rgba(11,14,20,.9);backdrop-filter:blur(8px);border-bottom:1px solid var(--border);padding:14px 20px;display:flex;align-items:center;gap:16px;justify-content:space-between}
.brand{font-weight:700;letter-spacing:.3px;display:flex;align-items:center;gap:10px}
.brand .dot{width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent)}
.badge{font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid var(--border);color:var(--muted)}
.container{max-width:1280px;margin:0 auto;padding:18px 20px}
.grid{display:grid;gap:14px}
.grid.kpi{grid-template-columns:repeat(4,1fr)}
@media(max-width:1100px){.grid.kpi{grid-template-columns:repeat(2,1fr)}}
@media(max-width:640px){.grid.kpi{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}
.card h3{margin:0 0 6px 0;font-size:12px;letter-spacing:.4px;color:var(--muted);text-transform:uppercase}
.value{font-size:22px;font-weight:700}.sub{font-size:12px;color:var(--muted)}
.row{display:flex;gap:14px;flex-wrap:wrap}
.row > *{flex:1 1 320px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 6px;border-bottom:1px solid var(--border);text-align:right}
th{color:var(--muted);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.4px}
th:first-child,td:first-child{text-align:left}
.bid{color:var(--accent)} .ask{color:var(--danger)}
.pill{padding:2px 8px;border-radius:999px;font-size:11px;border:1px solid var(--border)}
.pill.ok{color:var(--accent);border-color:rgba(110,231,183,.4)} .pill.warn{color:var(--warn)} .pill.danger{color:var(--danger)}
.spark{height:46px;background:linear-gradient(180deg,rgba(96,165,250,.18),transparent);border-radius:8px;border:1px solid var(--border)}
.footer{padding:18px;color:var(--muted);font-size:12px;text-align:center}
.kv{display:flex;justify-content:space-between;font-size:13px;padding:6px 0;border-bottom:1px dashed var(--border)}
.kv:last-child{border:none}
.controls{display:flex;gap:8px;flex-wrap:wrap}
.btn{padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:#0f1420;color:var(--text);cursor:pointer;font-size:13px}
.btn.primary{background:var(--accent2);border-color:var(--accent2);color:#081018;font-weight:700}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
</style>
</head>
<body>
<div class="header">
  <div class="brand"><span class="dot"></span> RL Market Maker <span class="badge">ENV: simulation • BTCUSDT • Testnet-safe</span></div>
  <div class="controls">
    <span class="pill ok" id="agentStatus">● Agent idle</span>
    <span class="pill" id="riskStatus">Risk OK</span>
    <a class="btn" href="/docs">API Docs</a>
    <a class="btn primary" href="/health">Health</a>
  </div>
</div>

<div class="container">
  <div class="grid kpi">
    <div class="card">
      <h3>Mid Price</h3>
      <div class="value mono" id="mid">—</div>
      <div class="sub">Best Bid <span id="bid" class="bid">—</span> • Ask <span id="ask" class="ask">—</span></div>
      <div class="sub">Spread <span id="spread">—</span> bps • <span id="midDelta">—</span></div>
    </div>
    <div class="card">
      <h3>Inventory</h3>
      <div class="value mono" id="inventory">—</div>
      <div class="sub">Max 1.0 • Ratio <span id="invRatio">—</span> • <span id="invState">—</span></div>
      <div class="spark" id="invSpark" title="inventory sparkline"></div>
    </div>
    <div class="card">
      <h3>PnL</h3>
      <div class="value mono" id="pnl">—</div>
      <div class="sub">Realized <span id="realized">—</span> • Unrealized <span id="unrealized">—</span></div>
      <div class="sub">Fees <span id="fees">—</span> • Sharpe <span id="sharpe">—</span></div>
    </div>
    <div class="card">
      <h3>RL Action & Reward</h3>
      <div class="value mono" id="rlAction">—</div>
      <div class="sub">Reward <span id="reward">—</span> • Q-Bias <span id="skew">—</span></div>
      <div class="sub">Policy <span id="policy">PPO</span> • Step <span id="step">—</span></div>
    </div>
  </div>

  <div class="row" style="margin-top:14px">
    <div class="card">
      <h3>Market Depth (top 10)</h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div>
          <table><thead><tr><th>Bid Price</th><th>Qty</th><th>Cum</th></tr></thead><tbody id="bids"></tbody></table>
        </div>
        <div>
          <table><thead><tr><th>Ask Price</th><th>Qty</th><th>Cum</th></tr></thead><tbody id="asks"></tbody></table>
        </div>
      </div>
      <div class="sub" style="margin-top:8px">Imbalance (10) <span id="imbalance">—</span> • Microprice <span id="micro">—</span></div>
    </div>

    <div class="card">
      <h3>Orders & Execution</h3>
      <div class="kv"><span>Open Orders</span><span class="mono" id="openOrders">—</span></div>
      <div class="kv"><span>Fill Rate</span><span class="mono" id="fillRate">—</span></div>
      <div class="kv"><span>Spread Captured</span><span class="mono" id="spreadCap">—</span></div>
      <div class="kv"><span>Cancels / min</span><span class="mono" id="cancels">—</span></div>
      <div class="kv"><span>Adverse Selection</span><span class="mono" id="adverse">—</span></div>
      <div class="sub" style="margin-top:8px">Quote Manager: <span id="quoteInfo">simulation • queue-aware</span></div>
    </div>
  </div>

  <div class="row" style="margin-top:14px">
    <div class="card">
      <h3>Equity & Inventory (mock)</h3>
      <canvas id="equity" width="600" height="140" style="width:100%;height:140px;background:#0f1420;border-radius:10px;border:1px solid var(--border)"></canvas>
      <div class="sub">Drawdown <span id="dd">—</span> • Updates every 1s from <code>/status</code></div>
    </div>
    <div class="card">
      <h3>Logs</h3>
      <div id="logs" class="mono" style="height:140px;overflow:auto;background:#0f1420;border:1px solid var(--border);border-radius:10px;padding:8px;font-size:12px;line-height:1.4"></div>
    </div>
  </div>

  <div class="footer">API: <code>/status</code> • <code>/health</code> • <code>/</code> — UI polls <code>/status</code> every 1s. Wire to <code>SimulatedExchange</code>/<code>BinanceTestnetExchange</code> in <code>src/market_maker/api/main.py:status()</code>.</div>
</div>

<script>
const $ = id => document.getElementById(id);
let eq=[], inv=[], logs=[];
function fmt(n,d=2){ if(n===null||n===undefined||isNaN(n)) return '—'; return Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}); }
function pushLog(m){ const t=new Date().toLocaleTimeString(); logs.unshift(`[${t}] ${m}`); if(logs.length>40) logs.pop(); $('logs').innerHTML=logs.map(l=>`<div>${l}</div>`).join(''); }
async function update(){
  try{
    const r=await fetch('/status'); const j=await r.json();
    // KPIs
    $('mid').textContent = j.mid ? fmt(j.mid,2) : '—';
    $('bid').textContent = j.bid ? fmt(j.bid,2) : '—';
    $('ask').textContent = j.ask ? fmt(j.ask,2) : '—';
    $('spread').textContent = j.spread_bps!=null? fmt(j.spread_bps,1): '—';
    $('midDelta').textContent = j.mid_change!=null? (j.mid_change>0? '+'+fmt(j.mid_change,2): fmt(j.mid_change,2)) : '—';
    $('inventory').textContent = j.inventory!=null? fmt(j.inventory,4): '—';
    $('invRatio').textContent = j.inventory_ratio!=null? fmt(j.inventory_ratio,2): '—';
    $('invState').textContent = j.inventory_state||'—';
    $('pnl').textContent = j.pnl!=null? (j.pnl>=0? '+'+fmt(j.pnl,2): fmt(j.pnl,2)) : '—';
    $('realized').textContent = j.realized!=null? fmt(j.realized,2): '—';
    $('unrealized').textContent = j.unrealized!=null? fmt(j.unrealized,2): '—';
    $('fees').textContent = j.fees!=null? fmt(j.fees,2): '0.00';
    $('sharpe').textContent = j.sharpe!=null? fmt(j.sharpe,2): '—';
    $('rlAction').textContent = j.rl_action? j.rl_action : (j.bid_offset!=null? `bid ${fmt(j.bid_offset,1)}bps / ask ${fmt(j.ask_offset,1)}bps` : '—');
    $('reward').textContent = j.reward!=null? fmt(j.reward,3): '—';
    $('skew').textContent = j.skew!=null? fmt(j.skew,2): '—';
    $('step').textContent = j.step!=null? j.step : '—';
    $('openOrders').textContent = j.open_orders!=null? j.open_orders : '—';
    $('fillRate').textContent = j.fill_rate!=null? fmt(j.fill_rate*100,1)+'%' : '—';
    $('spreadCap').textContent = j.spread_captured!=null? fmt(j.spread_captured,2): '—';
    $('cancels').textContent = j.cancels!=null? j.cancels : '—';
    $('adverse').textContent = j.adverse!=null? fmt(j.adverse,3): '—';
    $('imbalance').textContent = j.imbalance!=null? fmt(j.imbalance,2): '—';
    $('micro').textContent = j.microprice!=null? fmt(j.microprice,2): '—';
    $('dd').textContent = j.drawdown!=null? fmt(j.drawdown*100,2)+'%': '—';
    // depth
    if(j.bids){ $('bids').innerHTML = j.bids.slice(0,10).map((r,i)=>`<tr><td class="bid">${fmt(r[0],2)}</td><td>${fmt(r[1],4)}</td><td>${fmt(j.bids.slice(0,i+1).reduce((a,c)=>a+c[1],0),4)}</td></tr>`).join(''); }
    if(j.asks){ $('asks').innerHTML = j.asks.slice(0,10).map((r,i)=>`<tr><td class="ask">${fmt(r[0],2)}</td><td>${fmt(r[1],4)}</td><td>${fmt(j.asks.slice(0,i+1).reduce((a,c)=>a+c[1],0),4)}</td></tr>`).join(''); }
    // status pills
    $('agentStatus').textContent = j.agent_status||'● Agent idle';
    $('riskStatus').textContent = j.risk_status||'Risk OK';
    $('riskStatus').className = 'pill ' + (j.risk_status && j.risk_status.includes('LIMIT') ? 'danger' : 'ok');
    // sparklines (equity)
    if(j.pnl!=null){ eq.push(j.pnl); if(eq.length>60) eq.shift(); draw($('equity'), eq); }
    pushLog(`mid ${fmt(j.mid,2)} inv ${fmt(j.inventory,4)} pnl ${fmt(j.pnl,2)} ${j.risk_status||''}`);
  }catch(e){ pushLog('status fetch failed: '+e.message); }
}
function draw(canvas, data){
  const ctx=canvas.getContext('2d'); const w=canvas.width=canvas.clientWidth*2, h=canvas.height=140*2;
  ctx.clearRect(0,0,w,h);
  if(data.length<2) return;
  const min=Math.min(...data), max=Math.max(...data), rng=(max-min)||1;
  ctx.strokeStyle='#60a5fa'; ctx.lineWidth=2; ctx.beginPath();
  data.forEach((v,i)=>{ const x=i/(data.length-1)*w, y=h - ( (v-min)/rng*h*0.8 + h*0.1 ); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.stroke();
  // fill
  ctx.lineTo(w,h); ctx.lineTo(0,h); ctx.closePath(); ctx.fillStyle='rgba(96,165,250,.18)'; ctx.fill();
}
setInterval(update,1000); update();
pushLog('Dashboard loaded — polling /status every 1s');
</script>
</body>
</html>
    """

@app.get("/status")
def status():
    # Stub status - in production query SimulatedExchange / BinanceTestnetExchange and env
    # Simulated live-ish values for demo
    mid = 50000 + random.uniform(-150,150)
    spread_bps = random.uniform(6,18)
    bid = mid * (1 - spread_bps/10000/2)
    ask = mid * (1 + spread_bps/10000/2)
    inv = random.uniform(-0.4,0.4)
    pnl = random.uniform(-120,180)
    # depth
    bids = [[bid - i*0.5, random.uniform(0.3,4)] for i in range(10)]
    asks = [[ask + i*0.5, random.uniform(0.3,4)] for i in range(10)]
    return {
        "mid": round(mid,2),
        "bid": round(bid,2),
        "ask": round(ask,2),
        "spread_bps": round(spread_bps,2),
        "mid_change": round(random.uniform(-2,2),2),
        "microprice": round((bid*asks[0][1]+ask*bids[0][1])/(bids[0][1]+asks[0][1]),2),
        "imbalance": round((sum(b[1] for b in bids[:5])-sum(a[1] for a in asks[:5]))/(sum(b[1] for b in bids[:5])+sum(a[1] for a in asks[:5])),3),
        "inventory": round(inv,4),
        "inventory_ratio": round(inv/1.0,3),
        "inventory_state": "OK" if abs(inv)<0.8 else "NEAR LIMIT",
        "pnl": round(pnl,2),
        "realized": round(pnl*0.7,2),
        "unrealized": round(pnl*0.3,2),
        "fees": round(random.uniform(0,5),2),
        "sharpe": round(random.uniform(-0.5,1.5),2),
        "drawdown": round(random.uniform(0,0.02),4),
        "rl_action": f"bid {random.randint(-15,15)}bps / ask {random.randint(-15,15)}bps",
        "bid_offset": round(random.uniform(-20,20),1),
        "ask_offset": round(random.uniform(-20,20),1),
        "skew": round(random.uniform(-1,1),2),
        "reward": round(random.uniform(-1,1),3),
        "step": random.randint(100,900),
        "agent_status": "● Agent live (simulation)",
        "risk_status": "Risk OK" if abs(inv)<0.7 else "RISK LIMIT",
        "open_orders": random.randint(0,8),
        "fill_rate": round(random.uniform(0.45,0.85),2),
        "spread_captured": round(random.uniform(2,12),1),
        "cancels": random.randint(0,12),
        "adverse": round(random.uniform(-0.2,0.2),3),
        "bids": bids,
        "asks": asks,
    }

def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
