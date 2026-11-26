"""CRANK signal bot: USDJPY / GBPJPY (1h).  GOLD BOTIDAN MUSTAQIL (hech narsa import qilmaydi).

Eng yaxshi timeframe (backtest, 13-16 yil):
  USDJPY  H1, filtrsiz   (PF 1.15)
  GBPJPY  H1 + 4h filtr  (PF 1.29)
Chiqish faqat teskari signalda (SL/TP yo'q).  Holat: state_multi.json (gold state.json ga tegmaydi).
"""
import os, json, time, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from numpy.lib.stride_tricks import sliding_window_view

TG_TOKEN = os.environ["TG_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
TD_KEY = os.environ.get("TD_KEY")            # faqat src="td" ishlar uchun kerak (hozir hech biri ishlatmaydi)
STATE_FILE = os.environ.get("STATE_FILE", "state_multi.json")

# name, src ("yf" = Yahoo, kalit kerak emas | "td" = TwelveData), sym, tf (minut), htf (filtr, minut yoki None),
# unit: ("pip", 0.01) yoki ("punkt", 1.0), dec: narx kasr xonalari, long/short: yo'nalishni yoqish/o'chirish
JOBS = [
    dict(name="USDJPY", src="yf", sym="USDJPY=X", tf=60, htf=None, unit=("pip", 0.01), dec=3, long=True, short=True),
    dict(name="GBPJPY", src="yf", sym="GBPJPY=X", tf=60, htf=240, unit=("pip", 0.01), dec=3, long=True, short=True),
]

CRANK = dict(fast=11, slow=28, basis=56, norm_len=56, mult=1.6, thr=0.2,
             use_cci=True, cci_len=55, cci_up=120.0, cci_dn=-130.0)   # hajm ta'siri nol (tekshirilgan)
HTF_EMA = 50
MAX_LATE = 3           # signal kechiksa ham kirish faqat shu necha sham ichida
MIN_BARS = 500
MIN_HTF_BARS = 120
BARS = 5000
NOTIFY_SKIPPED = True
FMT = "%Y-%m-%d %H:%M:%S"
TFL = {15: "M15", 30: "M30", 60: "H1", 240: "H4"}

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

# ------------------------------------------------------------------ ma'lumot
def get_yf(sym, tf):
    import yfinance as yf
    interval = {15: "15m", 30: "30m", 60: "60m"}[tf]
    period = "730d" if tf >= 60 else "59d"
    last = None
    for a in range(3):
        try:
            d = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.dropna(subset=["Open", "High", "Low", "Close"])
            if len(d) == 0:
                raise Exception("bo'sh javob")
            idx = pd.to_datetime(d.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            d = d.copy(); d.index = idx
            d = d[~d.index.duplicated(keep="last")].sort_index()
            return pd.DataFrame({"time": d.index.strftime(FMT), "Open": d.Open.values, "High": d.High.values,
                                 "Low": d.Low.values, "Close": d.Close.values}).tail(BARS).reset_index(drop=True)
        except Exception as e:
            last = e
            time.sleep(3 * (a + 1))
    raise Exception(f"Yahoo xatosi ({sym}): {str(last)[:120]}")

def get_td(sym, tf):
    iv = {15: "15min", 30: "30min", 60: "1h", 240: "4h"}[tf]
    r = requests.get("https://api.twelvedata.com/time_series", timeout=20, params={
        "symbol": sym, "interval": iv, "outputsize": BARS, "timezone": "UTC", "apikey": TD_KEY}).json()
    if "values" not in r:
        raise Exception(f"Twelve Data xatosi: {r}")
    rows = [(v["datetime"], float(v["open"]), float(v["high"]), float(v["low"]), float(v["close"])) for v in reversed(r["values"])]
    return pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close"])

FETCH = {"yf": get_yf, "td": get_td}

# ------------------------------------------------------------------ CRANK
def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values

def _cci(x, n):
    w = sliding_window_view(x, n)
    sma = w.mean(1)
    md = np.abs(w - sma[:, None]).mean(1)
    out = np.full(len(x), np.nan)
    out[n - 1:] = (x[n - 1:] - sma) / (0.015 * md + 1e-12)
    return out

def compute_crank(close, p=CRANK):
    c = np.asarray(close, float)
    basis = _ema(c, p["basis"])
    ema_dir = np.where(_ema(c, p["fast"]) > _ema(c, p["slow"]), 1, -1)
    sd = pd.Series(c).rolling(p["norm_len"]).std(ddof=0).values
    norm = (c - basis) / (sd * p["mult"])
    cc = _cci(c, p["cci_len"])
    st = np.zeros(len(c), int)
    st[0] = ema_dir[0]
    for i in range(1, len(c)):
        if ema_dir[i] == 1 and norm[i] > p["thr"] and (not p["use_cci"] or cc[i] > p["cci_up"]):
            st[i] = 1
        elif ema_dir[i] == -1 and norm[i] < -p["thr"] and (not p["use_cci"] or cc[i] < p["cci_dn"]):
            st[i] = -1
        else:
            st[i] = st[i - 1]
    return st

def htf_name(m):
    return "1D" if m == 1440 else f"{m // 60}h"

def htf_dir_asof(cdf, tf, htf_min, n=HTF_EMA):
    """Shu sham yopilgan paytda ma'lum bo'lgan oxirgi TUGAGAN yuqori timeframe shamining yo'nalishi."""
    t = pd.to_datetime(cdf.time)
    c = pd.Series(cdf.Close.values, index=t)
    h = c.resample(f"{htf_min}min", label="left", closed="left").last().dropna()
    if len(h) < MIN_HTF_BARS:
        raise Exception(f"{htf_name(htf_min)} filtr uchun tarix yetarli emas: {len(h)} < {MIN_HTF_BARS}")
    ema = h.ewm(span=n, adjust=False).mean()
    known = pd.Series(np.where(h > ema, 1, -1), index=h.index + pd.Timedelta(minutes=htf_min))
    return known.reindex(t + pd.Timedelta(minutes=tf), method="ffill").fillna(0).astype(int).values

def _flips(st):
    return [i for i in range(1, len(st)) if st[i] != st[i - 1]]

def px(job, v):
    return f"{v:.{job['dec']}f}"

def fmt_pnl(job, pnl):
    u, s = job["unit"]
    return f"{pnl:+.{job['dec']}f} ({pnl / s:+.0f} pip)" if u == "pip" else f"{pnl:+.2f} punkt"

def send(text):
    print(text)
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                  data={"chat_id": CHAT_ID, "text": text}, timeout=15)

def label_of(job):
    return f"{job['name']} {TFL[job['tf']]} CRANK" + (f"+{htf_name(job['htf'])}" if job["htf"] else "")

def crank_init(job, cdf):
    c = cdf.Close.values
    st = compute_crank(c)
    fl = _flips(st)
    f = fl[-1] if fl else len(st) - 1
    side = "LONG" if st[-1] == 1 else "SHORT"
    allowed = job["long"] if side == "LONG" else job["short"]
    txt = f"Kuzatuv boshlandi: {label_of(job)}\nJoriy CRANK trendi: {side}, signal narxi {px(job, c[f])}"
    pos, aligned = None, True
    if job["htf"]:
        hd = htf_dir_asof(cdf, job["tf"], job["htf"])
        aligned = hd[-1] == st[-1]
        txt += f"\n{htf_name(job['htf'])} trend filtri: {'LONG' if hd[-1] == 1 else 'SHORT'} -> " + ("mos, kuzatamiz" if aligned else "mos emas, kutamiz")
    if allowed and aligned:
        pos = dict(side=side, entry=float(c[f]), t=cdf.time.iloc[-1], ft=cdf.time.iloc[f])
    send(txt + "\nBu eski signal, yangi kirish EMAS. Keyingi yangi signal xabar qilinadi.")
    return pos, cdf.time.iloc[f]

def crank_step(job, cdf, pos, flip_t):
    c = cdf.Close.values
    st = compute_crank(c)
    n = len(st)
    fl = _flips(st)
    if not fl:
        return pos, flip_t
    f = fl[-1]
    ft = cdf.time.iloc[f]
    if flip_t is not None and ft <= flip_t:
        return pos, flip_t
    new_n = sum(1 for i in fl if cdf.time.iloc[i] > (flip_t or ""))
    target = "LONG" if st[f] == 1 else "SHORT"
    cp = float(c[f])
    late = n - 1 - f
    label = label_of(job)
    note = f"\n(kechikkan signal: {late} sham oldin)" if late > 0 else ""
    if new_n > 1:
        note += f"\n(diqqat: {new_n} ta signal o'tkazib yuborilgan, oxirgisi hisobga olindi)"
    if pos is not None and pos["side"] != target:
        pnl = (cp - pos["entry"]) if pos["side"] == "LONG" else (pos["entry"] - cp)
        send(f"{label}: CHIQISH {pos['side']} (teskari signal) @ {px(job, cp)}\nNatija: {fmt_pnl(job, pnl)}{note}")
        pos = None
    elif pos is not None:
        return pos, ft
    allowed = job["long"] if target == "LONG" else job["short"]
    aligned = True
    if job["htf"]:
        aligned = htf_dir_asof(cdf, job["tf"], job["htf"])[f] == st[f]
    if allowed and aligned and late <= MAX_LATE:
        pos = dict(side=target, entry=cp, t=cdf.time.iloc[-1], ft=ft)
        extra = f"\n{htf_name(job['htf'])} trend filtri: mos" if job["htf"] else ""
        send(f"{label}: {target} kirish @ {px(job, cp)}\nChiqish: teskari signalda (SL/TP yo'q){extra}{note}")
    elif NOTIFY_SKIPPED:
        why = ("bu yo'nalish o'chirilgan" if not allowed else
               f"filtr rad etdi ({htf_name(job['htf'])} trend teskari)" if not aligned else "juda kech qoldi")
        send(f"{label}: {target} signal o'tkazib yuborildi - {why}{note}")
    return pos, ft

# ------------------------------------------------------------------ ish
def run_job(S, job):
    tf = job["tf"]
    key = f"{job['name']}|{tf}m|CRANK" + (f"|h{job['htf']}" if job["htf"] else "")
    st = S.get(key)
    now = utcnow()
    if st is not None:
        if st.get("retry") and now < datetime.strptime(st["retry"], FMT):
            return
        if now < datetime.strptime(st["last_t"], FMT) + timedelta(minutes=2 * tf):
            return                       # yangi sham hali yopilmagan -> so'rov yubormaymiz
    df = FETCH[job["src"]](job["sym"], tf)
    df["complete"] = [datetime.strptime(t, FMT) + timedelta(minutes=tf) <= now for t in df.time]
    cdf = df[df.complete].reset_index(drop=True)
    if len(cdf) < MIN_BARS:
        raise Exception(f"Sham yetarli emas: {len(cdf)} < {MIN_BARS}")
    if st is None:
        pos0, flip0 = crank_init(job, cdf)
        st = {"pos": pos0, "last_t": cdf.time.iloc[-1], "retry": None, "flip_t": flip0}
    pos, last_t, flip_t = st["pos"], st["last_t"], st.get("flip_t")
    if cdf.time.iloc[-1] != last_t:
        last_t = cdf.time.iloc[-1]
        pos, flip_t = crank_step(job, cdf, pos, flip_t)
    newest = datetime.strptime(df.time.iloc[-1], FMT)
    retry = (now + timedelta(minutes=30)).strftime(FMT) if newest + timedelta(minutes=2 * tf) < now else None
    S[key] = {"pos": pos, "last_t": last_t, "retry": retry, "flip_t": flip_t}

def main():
    try:
        S = json.load(open(STATE_FILE))
    except Exception:
        S = {}
    errs = []
    for job in JOBS:
        try:
            run_job(S, job)
        except Exception as e:
            print(job["name"], "xato:", e)
            errs.append(f"{job['name']} {TFL[job['tf']]}: {str(e)[:150]}")
    msg = "\n".join(errs)
    if msg and msg != S.get("_err"):
        send("XATO (multi):\n" + msg)
    S["_err"] = msg
    json.dump(S, open(STATE_FILE, "w"))

if __name__ == "__main__":
    main()
