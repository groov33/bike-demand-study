# Bike demand assistant for the operations manager.
# Run from the project folder:  streamlit run app.py
import calendar
import json
import math
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ["hr", "workingday", "holiday", "weekday", "mnth", "yr", "season", "weather", "temp", "hum", "windspeed"]
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MON = "(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + ")"


@st.cache_resource
def load():
    hour = pd.read_csv("data/hour.csv")
    hour["date"] = pd.to_datetime(hour["dteday"], format="%d-%m-%Y")
    hour["timestamp"] = hour["date"] + pd.to_timedelta(hour["hr"], unit="h")
    pred = pd.read_csv("model/replay_predictions.csv", parse_dates=["timestamp", "date"])
    model = joblib.load("model/demand_model.joblib")
    info = json.load(open("model/model_info.json"))
    return hour, pred, model, info


hour, pred, model, info = load()
observed = hour.set_index("timestamp")["cnt"]
predicted = pred.set_index("timestamp")["predicted"]


# ---------- bikes needed (Little's law + assumptions) ----------
def bikes_needed(rides_per_hour, trip_min, reserve, broken):
    on_road = rides_per_hour * trip_min / 60
    return on_road, math.ceil(round(on_road * (1 + reserve) / (1 - broken), 6))


# ---------- what-if day for the model ----------
def season_of(d):
    md = (d.month, d.day)
    if md >= (12, 21) or md < (3, 21):
        return 1
    return 2 if md < (6, 21) else 3 if md < (9, 23) else 4


def typical_weather(month):
    m = hour[hour.mnth == month]
    return m.temp.median() * 41, m.hum.median() * 100, m.windspeed.median() * 67


def predict_day(date, weather, temp_c, hum, wind, holiday=False):
    date = pd.Timestamp(date)
    weekday = (date.weekday() + 1) % 7                       # dataset uses 0 = Sunday
    X = pd.DataFrame({"hr": range(24), "workingday": int(weekday not in (0, 6) and not holiday),
                      "holiday": int(holiday), "weekday": weekday, "mnth": date.month, "yr": 1,
                      "season": season_of(date), "weather": weather, "temp": temp_c / 41,
                      "hum": hum / 100, "windspeed": wind / 67})
    return pd.Series(np.clip(model.predict(X[FEATURES]), 0, None)), X.workingday[0]


# ---------- question parsing ----------
class Ask(Exception):
    pass


def find_date(q):
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", q):
        y, mo, d = m.groups()
    elif m := re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MON}\.?,?\s+(\d{{4}})\b", q):
        d, mo, y = m.group(1), MONTHS[m.group(2)], m.group(3)
    elif m := re.search(rf"\b{MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", q):
        mo, d, y = MONTHS[m.group(1)], m.group(2), m.group(3)
    elif m := re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", q):
        a, b, y = (int(x) for x in m.groups())
        if a <= 12 and b <= 12 and a != b:
            raise Ask(f"Is '{m.group(0)}' {a} {calendar.month_name[b]} or {b} {calendar.month_name[a]}? "
                      "Please write it like 2012-10-15 or 15 Oct 2012.")
        d, mo = (a, b) if a > 12 else (b, a)
    elif re.search(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MON}\b(?!\s+\d)", q):
        raise Ask("Which year? The data covers 2011 and 2012.")
    else:
        return None
    try:
        return pd.Timestamp(int(y), int(mo), int(d))
    except ValueError:
        raise Ask("That isn't a valid date.")


def find_hour(q):
    if m := re.search(r"\b(\d{1,2})(?::00)?\s*(am|pm)\b", q):
        return int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
    if m := re.search(r"\b(\d{1,2}):00\b", q):
        return int(m.group(1))
    if m := re.search(r"\bat (\d{1,2})\b(?!\s*(?:degrees|c\b))", q):
        raise Ask(f"Do you mean {m.group(1)} am or {m.group(1)} pm?")
    return None


def day_frame(date):
    idx = pd.date_range(date, periods=24, freq="h")
    return pd.DataFrame({"actual": observed.reindex(idx).values, "predicted": predicted.reindex(idx).values},
                        index=range(24))


def answer(q, trip, reserve, broken):
    q = q.lower().strip()
    if re.search(r"station|where|location|route|driver", q):
        return "The data only has citywide totals, no stations or locations, so I can't say where bikes run out.", None
    if re.search(r"today|tomorrow|next (week|month)", q):
        return "There's no live data here. I can answer for dates in 2011-2012, or a what-if like 'a rainy weekday in July'.", None

    date = find_date(q)
    wants_bikes = "fleet" in q or ("bike" in q and re.search(r"need|enough|require", q))

    # what-if: a month without a year plus weather / day type
    month = re.search(rf"\b(?:in|during|for)\s+{MON}\b(?!\s+\d{{4}})", q)
    if date is None and month:
        weather = 3 if re.search(r"rain|snow|wet|storm", q) else 2 if re.search(r"mist|cloud|fog", q) else 1
        weekend = bool(re.search(r"weekend|saturday|sunday|non-working", q))
        temp_m = re.search(r"(-?\d{1,2})\s*(?:degrees|°|c\b)", q)
        mo = MONTHS[month.group(1)]
        d = pd.Timestamp(2012, mo, 15)
        d += pd.Timedelta(days=((5 if weekend else 2) - d.weekday()) % 7)   # a Saturday or a Wednesday
        temp, hum, wind = typical_weather(mo)
        if temp_m:
            temp = float(temp_m.group(1))
        prof, _ = predict_day(d, weather, temp, hum, wind)
        on_road, need = bikes_needed(prof.max(), trip, reserve, broken)
        w = {1: "clear", 2: "cloudy", 3: "rainy"}[weather]
        kind = "weekend day" if weekend else "weekday"
        return (f"**What-if: a {w} {kind} in {calendar.month_name[mo]}, {temp:.0f} °C** (model, 2012 demand level)\n\n"
                f"- expected rides: **{prof.sum():,.0f}**\n"
                f"- busiest hour: **{prof.idxmax()}:00** with about {prof.max():,.0f} rides\n"
                f"- bikes needed: **{need}** ({on_road:.0f} on the road at the peak)"), prof.rename("predicted")

    if date is None:
        if wants_bikes and (m := re.search(rf"\b{MON}\s+(\d{{4}})\b", q)):
            start = pd.Timestamp(int(m.group(2)), MONTHS[m.group(1)], 1)
            end = start + pd.offsets.MonthBegin(1)
            src = predicted if start in pred.date.values else observed
            s = src[(src.index >= start) & (src.index < end)]
            on_road, need = bikes_needed(s.max(), trip, reserve, broken)
            return (f"**{start:%B %Y}: about {need} bikes needed.** Busiest hour was {s.idxmax():%d %b %H:00} "
                    f"with {s.max():,.0f} rides ({on_road:.0f} bikes on the road)."), None
        if re.search(r"weekend|weekday|working", q):
            prof = hour.groupby(["hr", "workingday"]).cnt.mean().unstack().rename(columns={0: "non-working", 1: "working"})
            return ("Working days peak at 8am and 5-6pm (commuters). Weekends have one broad midday peak. "
                    "About 52% of a working day's rides fall in 7-9am and 5-7pm, vs 26% on weekends."), prof
        return ("Try something like: *How many bikes do we need on 2012-10-10?*, *What was the busiest hour on "
                "2012-11-07?*, or *How many bikes on a rainy weekday in July?*"), None

    if not hour.date.min() <= date <= hour.date.max():
        return "That date is outside the data (2011-01-01 to 2012-12-31).", None
    df = day_frame(date)
    has_pred = df.predicted.notna().any()
    col = "predicted" if has_pred else "actual"

    if wants_bikes:
        on_road, need = bikes_needed(df[col].max(), trip, reserve, broken)
        return (f"**{date:%a %d %b %Y}: about {need} bikes needed.**\n\n"
                f"Busiest hour {df[col].idxmax()}:00 with {df[col].max():,.0f} {col} rides → "
                f"{df[col].max():,.0f} × {trip} / 60 = {on_road:.0f} on the road, +{reserve:.0%} reserve, "
                f"÷ (1 − {broken:.0%}) for broken bikes."), df
    hr = find_hour(q)
    if hr is not None:
        a, p = df.actual[hr], df.predicted[hr]
        if np.isnan(a):
            return f"No recorded data for {date:%d %b %Y} at {hr}:00 (missing hour, not counted as zero).", df
        text = f"{date:%d %b %Y} at {hr}:00: **{a:,.0f} actual rides**"
        return (text + (f", {p:,.0f} predicted." if has_pred else ".")), df
    if re.search(r"busiest|peak|highest", q):
        text = f"Busiest hour on {date:%d %b %Y}: {df.actual.idxmax()}:00 with {df.actual.max():,.0f} actual rides"
        if has_pred:
            text += f" (predicted: {df.predicted.idxmax()}:00 with {df.predicted.max():,.0f})"
        return text + ".", df
    text = f"{date:%d %b %Y}: **{df.actual.sum():,.0f} actual rides**"
    return (text + (f", {df.predicted.sum():,.0f} predicted." if has_pred else ".")), df


# ---------- page ----------
st.set_page_config(page_title="Bike demand assistant", page_icon="🚲", layout="wide")
st.title("🚲 Bike demand assistant")
st.caption(f"Data 2011-2012. Predictions for Jul-Dec 2012 are month-ahead (model trained only on earlier data). "
           f"Model: {info['model']}, test error {info['test']['WAPE_%']:.1f}% WAPE.")

st.sidebar.header("Assumptions")
trip = st.sidebar.slider("Average trip (minutes)", 5, 60, 20)
reserve = st.sidebar.slider("Reserve on top of bikes in use (%)", 0, 150, 25, 5) / 100
broken = st.sidebar.slider("Bikes in repair (%)", 0, 40, 10) / 100
st.sidebar.caption("These are assumptions, not data. The fleet in the case is 2,000 bikes.")

ask_tab, whatif_tab = st.tabs(["Ask a question", "What-if planner"])

with ask_tab:
    examples = ["How many bikes do we need on 2012-10-10?", "What was the busiest hour on 2012-11-07?",
                "How many rentals on 15 October 2012 at 8am?", "How many bikes do we need in September 2012?",
                "How many bikes do we need on a rainy weekday in July?", "How do weekdays compare with weekends?"]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, use_container_width=True):
            st.session_state["q"] = ex
    q = st.text_input("Your question", key="q")
    if q:
        try:
            text, chart = answer(q, trip, reserve, broken)
            st.markdown(text)
            if chart is not None:
                st.line_chart(chart)
        except Ask as e:
            st.info(str(e))

with whatif_tab:
    c1, c2, c3 = st.columns(3)
    d = pd.Timestamp(c1.date_input("Date", pd.Timestamp("2012-07-18")))
    holiday = c1.checkbox("Public holiday")
    weather = c2.selectbox("Weather", [1, 2, 3], format_func={1: "clear", 2: "mist / cloudy", 3: "rain / snow"}.get)
    t0, h0, w0 = typical_weather(d.month)
    temp = c2.slider("Temperature (°C)", -8, 41, int(round(t0)))
    hum = c3.slider("Humidity (%)", 0, 100, int(round(h0)))
    growth = c3.slider("Demand growth vs 2012 (%)", -50, 100, 0, 5) / 100
    prof, wd = predict_day(d, weather, temp, hum, w0, holiday)
    prof = prof * (1 + growth)
    on_road, need = bikes_needed(prof.max(), trip, reserve, broken)
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected rides", f"{prof.sum():,.0f}", "working day" if wd else "non-working day", delta_color="off")
    m2.metric("Busiest hour", f"{prof.idxmax()}:00", f"{prof.max():,.0f} rides", delta_color="off")
    m3.metric("Bikes needed", need, f"{need - 2000:+,} vs 2,000 fleet", delta_color="off")
    st.line_chart(prof.rename("predicted rides"))
