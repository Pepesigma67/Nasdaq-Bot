"""Nasdaq Day-Trading-Bot (Paper). Ersetzt nasdaq_bot.py.

  python nasdaq_daytrader.py research  -> Daytrading-Analyse (15-Minuten-Kurse 2016-2026) und danach
                                          Tagesbasis-Analyse (2010-2026), jeweils mit Zufallstest
  python nasdaq_daytrader.py research-daily -> nur die Tagesbasis-Analyse
  python nasdaq_daytrader.py run       -> handelt die Gewinner automatisch im Alpaca-PAPER-Konto
                                          (Option --until MINUTEN beendet den Lauf nach dieser Zeit)

Installation:  pip install pandas numpy alpaca-py yfinance
Zugang:        Umgebungsvariablen ALPACA_KEY und ALPACA_SECRET (Paper-Keys von alpaca.markets)
"""
import json
import os
import pickle
import sys
import time
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST", "NFLX",
            "AMD", "ADBE", "PEP", "CSCO", "INTC", "QCOM", "TXN", "AMAT", "INTU", "BKNG",
            "ISRG", "MU", "LRCX", "PANW", "ADI", "KLAC", "SBUX", "GILD", "MDLZ", "REGN"]
START = datetime(2016, 1, 1, tzinfo=timezone.utc)
SPLIT = date(2022, 1, 1)      # 2016-2021: Parameter wählen, 2022-2026: ungesehen prüfen
COST = 0.0006                 # 0,06 % Kosten und Slippage pro Trade (hin und zurück)
N_RANDOM = 200                # Zufallsläufe pro Strategie
P_MAX = 0.0125                # strenge Schwelle gegen Zufallstreffer
MIN_MEAN_R = 0.05             # nach Kosten mindestens +0,05 R pro Trade
TOP_N = 3                     # der Bot nutzt nur die besten N bestandenen Strategien
RESULT = "passed_daytrade.json"
CACHE = "bars_cache.pkl"

RISK_EUR = 100                # Risiko pro Trade bis zum Stop-Loss
LOSS_LIMIT_EUR = 1000         # Tag beenden bei -1000 EUR
PROFIT_TARGET_EUR = 1000      # Tag beenden bei +1000 EUR
MAX_POS = 3                   # maximal gleichzeitige Positionen
MAX_POS_PCT = 0.25            # maximal 25 % des Depots pro Position
MIN_SP, MAX_SP = 0.002, 0.03  # erlaubter Stop-Abstand: 0,2 % bis 3 %
NY = "America/New_York"


# ---------- Daten ----------
def _clients():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"])


def fetch_history():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    cl, raw = _clients(), {}
    for s in UNIVERSE:
        print("Lade", s, flush=True)
        df = cl.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=s, timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=START, feed=DataFeed.IEX)).df
        raw[s] = df.droplevel(0)
    pickle.dump(raw, open(CACHE, "wb"))
    return raw


def build_days(raw):
    days = []
    for sym, df in raw.items():
        d = df.copy()
        d.index = d.index.tz_convert(NY)
        d = d.between_time("09:30", "15:45")
        prev = None
        for dt_, g in d.groupby(d.index.date):
            c = g["close"].values
            if prev is not None and len(g) >= 20:
                days.append(SimpleNamespace(
                    sym=sym, date=dt_, o=g["open"].values, h=g["high"].values, l=g["low"].values,
                    c=c, v=g["volume"].values, prev=prev))
            prev = c[-1]
    return days


# ---------- Signale: liefern (Einstiegs-Index, Stop-Abstand) oder None ----------
def _sp(entry, stop):
    sp = (entry - stop) / entry
    return sp if MIN_SP <= sp <= MAX_SP else None


def sig_orb(d, n):
    """Ausbruch über die Spanne der ersten n Kerzen."""
    if len(d.c) <= n:
        return None
    hi, lo = d.h[:n].max(), d.l[:n].min()
    for k in range(n, len(d.c)):
        if d.c[k] > hi:
            sp = _sp(d.c[k], lo)
            return (k + 1, sp) if sp else None
    return None


def sig_gap(d, gap):
    """Aufwärtslücke zur Eröffnung, erste Kerze grün."""
    if d.o[0] / d.prev - 1 < gap or d.c[0] <= d.o[0]:
        return None
    sp = _sp(d.c[0], d.l[0])
    return (1, sp) if sp else None


def sig_vwap(d):
    """Kurs steigt im positiven Tag wieder über den VWAP."""
    tp = (d.h + d.l + d.c) / 3
    vw = np.cumsum(tp * d.v) / np.maximum(np.cumsum(d.v), 1)
    for k in range(3, len(d.c)):
        if d.c[k - 1] < vw[k - 1] and d.c[k] > vw[k] and d.c[k] > d.prev:
            sp = _sp(d.c[k], d.l[k - 2:k + 1].min())
            return (k + 1, sp) if sp else None
    return None


# letzter Parameter jeder Zeile ist das Kursziel in Vielfachen des Risikos (R)
STRATS = {
    "orb": (sig_orb, [(n, t) for n in (1, 2, 4) for t in (1, 2, 3)]),
    "gap_go": (sig_gap, [(g, t) for g in (0.005, 0.01, 0.02) for t in (1, 2, 3)]),
    "vwap_reclaim": (sig_vwap, [(t,) for t in (1, 2, 3)]),
}


# ---------- Simulation ----------
def simulate(d, i, sp, tgt):
    """Einstieg zum Open der Kerze i, Stop/Ziel, sonst Ausstieg zum Tagesende. Ergebnis in R.
    Wenn Stop und Ziel in derselben Kerze liegen, gilt vorsichtig der Stop."""
    entry = d.o[i]
    stop, take = entry * (1 - sp), entry * (1 + sp * tgt)
    for j in range(i, len(d.c)):
        if d.l[j] <= stop:
            return -1.0 - COST / sp
        if d.h[j] >= take:
            return tgt - COST / sp
    return (d.c[-1] - entry) / entry / sp - COST / sp


def trades_for(days, fn, p):
    out = []
    for di, d in enumerate(days):
        s = fn(d, *p[:-1])
        if s and s[0] < len(d.c):
            out.append((di, s[0], s[1], simulate(d, s[0], s[1], p[-1])))
    return out


def random_test(days, trades, tgt, rng):
    """Münzwurf-Test: gleiche Tage, gleiche Aktien, gleicher Stop-Abstand, aber zufällige Einstiegszeit.
    p = Anteil der Zufallsläufe, die im Schnitt gleich gut oder besser sind."""
    if len(trades) > 2000:
        trades = [trades[i] for i in rng.choice(len(trades), 2000, replace=False)]
    real = float(np.mean([t[3] for t in trades]))
    rand = []
    for _ in range(N_RANDOM):
        rs = [simulate(days[di], int(rng.integers(1, len(days[di].c) - 1)), sp, tgt)
              for di, _e, sp, _r in trades]
        rand.append(np.mean(rs))
    return real, float((np.array(rand) >= real).mean())


def research(days=None):
    days = build_days(fetch_history()) if days is None else days
    rng = np.random.default_rng(42)
    print(f"{len(days)} Handelstage-Aktien-Kombinationen geladen\n")
    passed = []
    for name, (fn, grid) in STRATS.items():
        tr = {p: trades_for(days, fn, p) for p in grid}
        part = lambda t, oos: [x for x in t if (days[x[0]].date >= SPLIT) == oos]
        cands = [(np.mean([x[3] for x in part(tr[p], False)]), p)
                 for p in grid if len(part(tr[p], False)) >= 200]
        if not cands:
            print(f"{name:13s} zu wenige Trades -> aussortiert")
            continue
        s_in, best = max(cands)
        oos = part(tr[best], True)
        if len(oos) < 100:
            print(f"{name:13s} zu wenige Trades im Prüfzeitraum -> aussortiert")
            continue
        real, p = random_test(days, oos, best[-1], rng)
        win = float(np.mean([x[3] > 0 for x in oos]))
        ok = p < P_MAX and real > MIN_MEAN_R
        print(f"{name:13s} Parameter {best}  R/Trade vorher {s_in:+.2f}  Prüfung {real:+.2f}  "
              f"Trefferquote {win:.0%}  Trades {len(oos)}  p={p:.4f}  -> "
              f"{'BESTANDEN' if ok else 'aussortiert'}")
        if ok:
            passed.append({"name": name, "params": list(best), "oos_mean_r": real, "p": p, "win_rate": win})
    passed = sorted(passed, key=lambda x: -x["oos_mean_r"])[:TOP_N]
    json.dump(passed, open(RESULT, "w"), indent=2)
    print(f"\n{len(passed)} Strategie(n) gespeichert in {RESULT}")
    return passed


# ---------- Tagesbasis-Analyse (2010-2026, swing statt Daytrading) ----------
SPLIT_D = date(2020, 1, 1)    # davor: Parameter wählen, danach: ungesehen prüfen
COST_D = 0.0005               # 0,05 % pro Positionswechsel (Gebühren und Kursabweichung)
RESULT_D = "passed_daily.json"


def sharpe(x):
    sd = x.std()
    return 0.0 if sd == 0 else float(np.sqrt(252) * x.mean() / sd)


def load_daily():
    syms = UNIVERSE + ["QQQ"]
    try:
        import yfinance as yf
        df = yf.download(syms, start="2010-01-01", auto_adjust=True, progress=False)
        close = df["Close"].dropna(how="all")
        if len(close) > 1000:
            print(f"Tageskurse von Yahoo: {len(close)} Tage, {close.shape[1]} Werte", flush=True)
            return close
        print("Yahoo lieferte zu wenig Daten, wechsle zu Alpaca (ab 2016)", flush=True)
    except Exception as e:
        print("Yahoo nicht erreichbar:", e, "-> wechsle zu Alpaca (ab 2016)", flush=True)
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    bars = _clients().get_stock_bars(StockBarsRequest(
        symbol_or_symbols=syms, timeframe=TimeFrame.Day, start=START,
        feed=DataFeed.IEX, adjustment=Adjustment.ALL)).df
    close = bars["close"].unstack(level=0)
    close.index = close.index.tz_convert(None).normalize()
    print(f"Tageskurse von Alpaca: {len(close)} Tage, {close.shape[1]} Werte", flush=True)
    return close


def _hold_df(c, on, off):
    pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    pos[on] = 1
    pos[off] = 0
    return pos.ffill().fillna(0) > 0


def daily_pos(c, kind, p):
    if kind == "sma_cross":
        return c.rolling(p[0]).mean() > c.rolling(p[1]).mean()
    if kind == "momentum":
        return c.pct_change(p[0]) > 0
    if kind == "breakout":
        hi = c.rolling(p[0]).max().shift(1)
        lo = c.rolling(max(p[0] // 2, 2)).min().shift(1)
        return _hold_df(c, c > hi, c < lo)
    d = c.diff()                                  # rsi_trend: Rücksetzer im Aufwärtstrend
    up = d.clip(lower=0).ewm(alpha=1 / p[0]).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / p[0]).mean()
    rsi = 100 - 100 / (1 + up / dn)
    return _hold_df(c, (rsi < p[1]) & (c > c.rolling(200).mean()), rsi > 60)


DAILY = {
    "sma_cross": [(f, s) for f in (10, 20, 50) for s in (100, 150, 200)],
    "momentum": [(n,) for n in (20, 60, 120, 250)],
    "breakout": [(n,) for n in (20, 50, 100)],
    "rsi_trend": [(n, l) for n in (2, 5) for l in (5, 10, 20)],
}


def _dret(P, R):
    net = P * R - np.abs(np.diff(P, axis=0, prepend=np.zeros((1, P.shape[1])))) * COST_D
    return net.mean(axis=1)          # gleich gewichtet über alle Aktien, Cash wenn nicht investiert


def research_daily(c=None, n_random=500):
    c = load_daily() if c is None else c
    rng = np.random.default_rng(7)
    R = c.pct_change().fillna(0).values
    ins = np.array([d.date() < SPLIT_D for d in c.index])
    bh = sharpe(R[~ins].mean(axis=1))
    print(f"\nTAGESBASIS: {c.shape[1]} Werte, Prüfzeitraum ab {SPLIT_D}, Buy & Hold Sharpe {bh:.2f}\n")
    passed = []
    for name, grid in DAILY.items():
        pos = {g: np.vstack([np.zeros((1, c.shape[1])), daily_pos(c, name, g).values[:-1].astype(float)])
               for g in grid}                       # Signal von gestern, Einstieg heute
        best = max(grid, key=lambda g: sharpe(_dret(pos[g][ins], R[ins])))
        s_in = sharpe(_dret(pos[best][ins], R[ins]))
        Pr, Rr = pos[best][~ins], R[~ins]
        s_out = sharpe(_dret(Pr, Rr))
        n = len(Pr)
        rand = np.array([sharpe(_dret(np.roll(Pr, int(rng.integers(20, n - 20)), axis=0), Rr))
                         for _ in range(n_random)])
        p = float((rand >= s_out).mean())
        invested = float(Pr.mean())
        ok = p < P_MAX and s_out > 0
        print(f"{name:10s} Parameter {best}  Sharpe vorher {s_in:5.2f}  Prüfung {s_out:5.2f}  "
              f"investiert {invested:.0%}  p={p:.4f}  -> {'BESTANDEN' if ok else 'aussortiert'}")
        if ok:
            passed.append({"name": name, "params": list(best), "oos_sharpe": s_out, "p": p, "bh_sharpe": bh})
    json.dump(sorted(passed, key=lambda x: -x["oos_sharpe"])[:TOP_N], open(RESULT_D, "w"), indent=2)
    print(f"\n{len(passed)} Tagesstrategie(n) gespeichert in {RESULT_D}")
    return passed


# ---------- Live (Alpaca Paper) ----------
def _frame(g, prev):
    return SimpleNamespace(o=g["open"].values, h=g["high"].values, l=g["low"].values,
                           c=g["close"].values, v=g["volume"].values, prev=prev)


def prev_closes(dc, today):
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    bars = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=UNIVERSE, timeframe=TimeFrame.Day,
                                              start=today - timedelta(days=10), feed=DataFeed.IEX)).df
    prev = {}
    for s in UNIVERSE:
        try:
            g = bars.loc[s]
            prev[s] = float(g[g.index.date < today]["close"].iloc[-1])
        except Exception:
            pass
    return prev


def scan(tc, dc, cfg, slot, prev, traded, equity, risk_usd):
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
    bars = dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=UNIVERSE, timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=slot.replace(hour=9, minute=30), feed=DataFeed.IEX)).df
    n_open = len(tc.get_all_positions())
    for sym in UNIVERSE:
        if sym in traded or n_open >= MAX_POS or sym not in prev:
            continue
        try:
            g = bars.loc[sym]
        except KeyError:
            continue
        g = g[g.index < slot]                      # nur abgeschlossene Kerzen
        if len(g) == 0:
            continue
        d = _frame(g, prev[sym])
        for s in cfg:
            sig = STRATS[s["name"]][0](d, *s["params"][:-1])
            if not sig or sig[0] != len(d.c):      # Signal muss genau auf der letzten Kerze entstanden sein
                continue
            sp, tgt, price = sig[1], s["params"][-1], float(d.c[-1])
            shares = min(int(risk_usd / (price * sp)), int(equity * MAX_POS_PCT / price))
            if shares < 1:
                continue
            tc.submit_order(MarketOrderRequest(
                symbol=sym, qty=shares, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(price * (1 + sp * tgt), 2)),
                stop_loss=StopLossRequest(stop_price=round(price * (1 - sp), 2))))
            print(f"{slot:%H:%M} KAUF {shares} {sym} @~{price:.2f} Stop {price*(1-sp):.2f} "
                  f"Ziel {price*(1+sp*tgt):.2f} ({s['name']})")
            traded.add(sym)
            n_open += 1
            break


def todays_traded(tc, today):
    """Aktien, mit denen heute schon gehandelt wurde (aus Alpaca gelesen, übersteht Neustarts)."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    start = pd.Timestamp(today).tz_localize(NY).to_pydatetime()
    orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=start, limit=500))
    return {o.symbol for o in orders}


def run(until_min=None):
    """Handelt bis Börsenschluss. Mit until_min beendet sich der Lauf nach so vielen Minuten
    (für GitHub: ein Auftrag darf höchstens 6 Stunden laufen). Der Zustand wird bei Alpaca gelesen,
    ein Neustart verliert also nichts."""
    from alpaca.trading.client import TradingClient
    cfg = json.load(open(RESULT))
    if not cfg:
        print("Keine Strategie hat den Test bestanden. Der Bot handelt nicht.")
        return
    tc = TradingClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], paper=True)
    dc = _clients()
    try:
        import yfinance as yf
        fx = float(yf.download("EURUSD=X", period="5d", progress=False)["Close"].squeeze().iloc[-1])
    except Exception:
        fx = 1.08
        print("Wechselkurs nicht abrufbar, nehme 1,08 an")
    loss_limit, profit_target, risk_usd = LOSS_LIMIT_EUR * fx, PROFIT_TARGET_EUR * fx, RISK_EUR * fx
    print(f"Strategien: {[c['name'] for c in cfg]}  Grenzen: -{loss_limit:.0f} / +{profit_target:.0f} USD", flush=True)
    t0 = time.time()
    state_day = last_slot = None
    prev = {}
    while True:
        try:
            if until_min and (time.time() - t0) / 60 >= until_min:
                print("Zeitlimit dieses Laufs erreicht, der nächste Lauf übernimmt")
                return
            clock = tc.get_clock()
            if not clock.is_open:
                wait = (pd.Timestamp(clock.next_open) - pd.Timestamp(clock.timestamp)).total_seconds() / 60
                if wait > 100:
                    print("Börse ist heute zu oder schon geschlossen. Ende.")
                    return
                time.sleep(60)
                continue
            now = pd.Timestamp(clock.timestamp).tz_convert(NY)
            today = now.date()
            if today != state_day:
                state_day, prev = today, prev_closes(dc, today)
            acc = tc.get_account()
            equity = float(acc.equity)
            pnl = equity - float(acc.last_equity)
            if pnl <= -loss_limit or pnl >= profit_target:   # ohne Merker: bleibt am Tag wahr
                if tc.get_all_positions():
                    tc.close_all_positions(cancel_orders=True)
                    print(f"{today}: Tagesgrenze erreicht (P&L {pnl:+.0f} USD), alles geschlossen", flush=True)
                time.sleep(600)
                continue
            minutes_to_close = (pd.Timestamp(clock.next_close) - pd.Timestamp(clock.timestamp)).total_seconds() / 60
            if minutes_to_close <= 10:                 # vor Handelsschluss alles glattstellen
                if tc.get_all_positions():
                    tc.close_all_positions(cancel_orders=True)
                    print(f"{today}: Tagesende, alles geschlossen, P&L {pnl:+.0f} USD", flush=True)
                time.sleep(60)
                continue
            slot = now.replace(minute=now.minute // 15 * 15, second=0, microsecond=0)
            if slot != last_slot and (now - slot).total_seconds() >= 10:
                scan(tc, dc, cfg, slot, prev, todays_traded(tc, today), equity, risk_usd)
                last_slot = slot
            time.sleep(5)
        except Exception as e:
            print("Fehler:", e, flush=True)
            time.sleep(30)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "research":
        research()
        research_daily()
    elif cmd == "research-daily":
        research_daily()
    elif cmd == "run":
        until = int(sys.argv[sys.argv.index("--until") + 1]) if "--until" in sys.argv else None
        run(until)
    else:
        print(__doc__)
