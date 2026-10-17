import os
import re
import html
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
import feedparser

st.set_page_config(
    page_title="AI Financial Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COMPANIES = {
    "Reliance Industries": {"ticker": "RELIANCE.NS", "sector": "Energy / Retail / Telecom"},
    "Tata Consultancy Services": {"ticker": "TCS.NS", "sector": "Information Technology"},
    "Infosys": {"ticker": "INFY.NS", "sector": "Information Technology"},
    "HDFC Bank": {"ticker": "HDFCBANK.NS", "sector": "Banking"},
    "ICICI Bank": {"ticker": "ICICIBANK.NS", "sector": "Banking"},
    "State Bank of India": {"ticker": "SBIN.NS", "sector": "Banking"},
    "Bharti Airtel": {"ticker": "BHARTIARTL.NS", "sector": "Telecommunications"},
    "Larsen & Toubro": {"ticker": "LT.NS", "sector": "Engineering / Infrastructure"},
    "ITC": {"ticker": "ITC.NS", "sector": "Consumer Goods"},
    "Tata Motors": {"ticker": "TATAMOTORS.NS", "sector": "Automotive"},
    "Wipro": {"ticker": "WIPRO.NS", "sector": "Information Technology"},
    "Hindustan Unilever": {"ticker": "HINDUNILVR.NS", "sector": "Consumer Goods"},
    "Axis Bank": {"ticker": "AXISBANK.NS", "sector": "Banking"},
    "Maruti Suzuki": {"ticker": "MARUTI.NS", "sector": "Automotive"},
    "Mahindra & Mahindra": {"ticker": "M&M.NS", "sector": "Automotive"},
    "Asian Paints": {"ticker": "ASIANPAINT.NS", "sector": "Consumer Goods"},
}
TICKER_TO_NAME = {v["ticker"]: k for k, v in COMPANIES.items()}
LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def get_secret(name, default=""):
    """Read a Streamlit secret or environment variable without requiring either."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if data is None or data.empty:
        return pd.DataFrame()
    data = data.reset_index()
    if "Date" not in data.columns:
        data = data.rename(columns={data.columns[0]: "Date"})
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    return data


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_company_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    try:
        info = stock.get_info() or {}
    except Exception:
        info = {}

    # Fallback: Yahoo Finance sometimes omits marketCap from get_info().
    # Estimate it for whichever company is selected using fast_info price × shares.
    try:
        fast = stock.fast_info
        fast_price = number_or_none(fast.get("last_price"))
        fast_shares = number_or_none(fast.get("shares"))
        if number_or_none(info.get("marketCap")) is None and fast_price and fast_shares:
            info["marketCap"] = fast_price * fast_shares
            info["_marketCapEstimated"] = True
    except Exception:
        pass

    def safe_statement(attr):
        try:
            value = getattr(stock, attr)
            return value if isinstance(value, pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return {
        "info": info,
        "income": safe_statement("income_stmt"),
        "balance": safe_statement("balance_sheet"),
        "cashflow": safe_statement("cashflow"),
    }


def number_or_none(value):
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def fmt_currency(value, currency="INR"):
    value = number_or_none(value)
    if value is None:
        return "N/A"
    if abs(value) >= 1e12:
        return f"₹{value/1e12:,.2f}T" if currency == "INR" else f"{value/1e12:,.2f}T"
    if abs(value) >= 1e9:
        return f"₹{value/1e9:,.2f}B" if currency == "INR" else f"{value/1e9:,.2f}B"
    if abs(value) >= 1e7:
        return f"₹{value/1e7:,.2f} Cr" if currency == "INR" else f"{value:,.0f}"
    return f"₹{value:,.2f}" if currency == "INR" else f"{value:,.2f}"


def statement_value(statement: pd.DataFrame, possible_names):
    if statement is None or statement.empty:
        return None
    try:
        index_map = {str(idx).strip().lower(): idx for idx in statement.index}
        for name in possible_names:
            key = name.lower()
            if key in index_map:
                row = statement.loc[index_map[key]]
                if isinstance(row, pd.Series):
                    vals = row.dropna()
                    if not vals.empty:
                        return number_or_none(vals.iloc[0])
    except Exception:
        pass
    return None


def get_financial_summary(company_data: dict) -> dict:
    info = company_data.get("info", {}) or {}
    income = company_data.get("income", pd.DataFrame())
    balance = company_data.get("balance", pd.DataFrame())
    revenue = number_or_none(info.get("totalRevenue"))
    if revenue is None:
        revenue = statement_value(income, ["Total Revenue", "Operating Revenue", "Revenue"])
    net_income = number_or_none(info.get("netIncomeToCommon"))
    if net_income is None:
        net_income = statement_value(income, ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"])
    total_debt = number_or_none(info.get("totalDebt"))
    if total_debt is None:
        total_debt = statement_value(balance, ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    total_cash = number_or_none(info.get("totalCash"))
    if total_cash is None:
        total_cash = statement_value(balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents And Short Term Investments"])
    return {
        "revenue": revenue,
        "net_income": net_income,
        "total_debt": total_debt,
        "total_cash": total_cash,
        "market_cap": number_or_none(info.get("marketCap")),
        "trailing_pe": number_or_none(info.get("trailingPE")),
        "forward_pe": number_or_none(info.get("forwardPE")),
        "price_to_book": number_or_none(info.get("priceToBook")),
        "profit_margin": number_or_none(info.get("profitMargins")),
        "operating_margin": number_or_none(info.get("operatingMargins")),
        "return_on_equity": number_or_none(info.get("returnOnEquity")),
        "current_ratio": number_or_none(info.get("currentRatio")),
        "quick_ratio": number_or_none(info.get("quickRatio")),
        "debt_to_equity": number_or_none(info.get("debtToEquity")),
        "dividend_yield": number_or_none(info.get("dividendYield")),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "currency": info.get("currency", "INR"),
        "website": info.get("website", ""),
        "long_name": info.get("longName") or info.get("shortName") or "",
        "summary": info.get("longBusinessSummary", ""),
    }


def newsapi_key():
    return get_secret("NEWSAPI_KEY", "")


def fetch_news(company_name: str, ticker: str, max_items: int = 20) -> pd.DataFrame:
    """Use NewsAPI when configured; otherwise try Google News RSS. Headlines only are sent to FinBERT."""
    api_key = newsapi_key()
    articles = []
    query = f'"{company_name}" OR "{ticker.replace(".NS", "")}"'
    if api_key:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": min(max_items, 100),
                    "apiKey": api_key,
                },
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") == "ok":
                for item in payload.get("articles", []):
                    articles.append({
                        "title": item.get("title") or "",
                        "description": item.get("description") or "",
                        "source": (item.get("source") or {}).get("name", "NewsAPI"),
                        "published_at": item.get("publishedAt", ""),
                        "url": item.get("url", ""),
                    })
        except Exception as exc:
            st.warning(f"NewsAPI could not be reached; trying RSS fallback. Details: {exc}")
    if not articles:
        rss_query = quote_plus(f'"{company_name}" stock OR financial results')
        rss_url = f"https://news.google.com/rss/search?q={rss_query}&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:max_items]:
                source = ""
                if getattr(entry, "source", None):
                    source = getattr(entry.source, "title", "")
                published = getattr(entry, "published", "")
                try:
                    published = datetime(*entry.published_parsed[:6]).isoformat() if entry.get("published_parsed") else published
                except Exception:
                    pass
                articles.append({
                    "title": re.sub(r"\s+", " ", getattr(entry, "title", "")).strip(),
                    "description": re.sub(r"<[^>]+>", " ", getattr(entry, "summary", "")).strip(),
                    "source": source or "Google News RSS",
                    "published_at": published,
                    "url": getattr(entry, "link", ""),
                })
        except Exception as exc:
            st.warning(f"RSS news feed could not be reached: {exc}")
    frame = pd.DataFrame(articles)
    if frame.empty:
        return pd.DataFrame(columns=["title", "description", "source", "published_at", "url"])
    frame = frame.drop_duplicates(subset=["title"]).head(max_items).reset_index(drop=True)
    frame["published_at"] = frame["published_at"].fillna("").astype(str)
    frame["title"] = frame["title"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["source"] = frame["source"].fillna("Unknown").astype(str)
    frame["url"] = frame["url"].fillna("").astype(str)
    return frame


@st.cache_resource(show_spinner="Loading FinBERT model (first run can take a few minutes)...")
def load_finbert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if __import__("torch").cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        max_length=256,
    )


def analyze_news(news: pd.DataFrame) -> pd.DataFrame:
    if news is None or news.empty:
        return pd.DataFrame()
    model = load_finbert()
    texts = []
    for _, row in news.iterrows():
        title = str(row.get("title", "")).strip()
        desc = str(row.get("description", "")).strip()
        # Bound input length and avoid passing blank strings to the model.
        texts.append((title + ". " + desc)[:1500] or title)
    predictions = model(texts, batch_size=8)
    labels, confidences, pos_values, neg_values, neu_values = [], [], [], [], []
    for prediction in predictions:
        if isinstance(prediction, dict):
            prediction = [prediction]
        probs = {str(item["label"]).lower(): float(item["score"]) for item in prediction}
        best = max(probs, key=probs.get)
        labels.append(best)
        confidences.append(probs[best])
        pos_values.append(probs.get("positive", 0.0))
        neg_values.append(probs.get("negative", 0.0))
        neu_values.append(probs.get("neutral", 0.0))
    result = news.copy()
    result["sentiment"] = labels
    result["confidence"] = confidences
    result["positive_prob"] = pos_values
    result["negative_prob"] = neg_values
    result["neutral_prob"] = neu_values
    result["sentiment_score"] = result["positive_prob"] - result["negative_prob"]
    return result


def calculate_market_metrics(history: pd.DataFrame) -> dict:
    if history is None or history.empty or "Close" not in history:
        return {}
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(close) < 2:
        return {}
    returns = close.pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else None
    running_max = close.cummax()
    drawdown = close / running_max - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None
    recent_return = float(close.iloc[-1] / close.iloc[max(0, len(close) - 22)] - 1) if len(close) > 1 else None
    year_return = float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else None
    sharpe = None
    if len(returns) > 1 and returns.std() != 0:
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))
    return {
        "latest_close": float(close.iloc[-1]),
        "annualized_volatility": vol,
        "max_drawdown": max_drawdown,
        "approx_1m_return": recent_return,
        "period_return": year_return,
        "sharpe_rf0": sharpe,
        "drawdown_series": drawdown,
        "returns": returns,
    }


def bounded(value, low=0.0, high=100.0):
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(value, low, high))


def calculate_risk(financials: dict, metrics: dict, news_scored: pd.DataFrame):
    """Transparent heuristic risk score; components are indicative, not a validated rating."""
    components = {}
    weights = {}
    vol = metrics.get("annualized_volatility")
    if vol is not None:
        components["Market volatility"] = bounded(vol * 150)
        weights["Market volatility"] = 0.25
    dd = metrics.get("max_drawdown")
    if dd is not None:
        components["Maximum drawdown"] = bounded(abs(min(dd, 0)) * 200)
        weights["Maximum drawdown"] = 0.20

    debt_eq = financials.get("debt_to_equity")
    if debt_eq is not None:
        # Yahoo's debtToEquity is commonly expressed as a percentage (e.g. 80 = 0.80x).
        ratio = debt_eq / 100.0 if debt_eq > 10 else debt_eq
        components["Debt burden"] = bounded((ratio / 2.0) * 100)
        weights["Debt burden"] = 0.15

    margin = financials.get("profit_margin")
    current_ratio = financials.get("current_ratio")
    fundamental_parts = []
    if margin is not None:
        fundamental_parts.append(100 if margin < 0 else bounded((0.20 - margin) / 0.20 * 50))
    if current_ratio is not None:
        fundamental_parts.append(100 if current_ratio < 0.75 else (50 if current_ratio < 1.0 else 0))
    if fundamental_parts:
        components["Fundamental pressure"] = float(np.mean(fundamental_parts))
        weights["Fundamental pressure"] = 0.20

    if news_scored is not None and not news_scored.empty and "negative_prob" in news_scored:
        neg_share = float((news_scored["sentiment"] == "negative").mean())
        mean_sentiment = float(news_scored["sentiment_score"].mean())
        sentiment_risk = bounded(neg_share * 70 + max(0, -mean_sentiment) * 30)
        components["News sentiment risk"] = sentiment_risk
        weights["News sentiment risk"] = 0.20

    valid = [(name, score, weights[name]) for name, score in components.items() if score is not None]
    if not valid:
        return None, components, "Insufficient data to calculate risk score."
    total_weight = sum(w for _, _, w in valid)
    overall = sum(score * w for _, score, w in valid) / total_weight
    coverage = total_weight / sum(weights.values()) if weights else 0
    note = f"{len(valid)} of 5 risk components available; approximate coverage {coverage:.0%}."
    return round(float(overall), 1), components, note


def risk_label(score):
    if score is None:
        return "Unavailable"
    if score < 30:
        return "Lower observed risk"
    if score < 60:
        return "Moderate observed risk"
    return "Higher observed risk"


def add_comma(num):
    if num is None:
        return "N/A"
    return f"{num:,.2f}"


def company_comparison(selected_tickers, period):
    rows = []
    for ticker in selected_tickers:
        name = TICKER_TO_NAME.get(ticker, ticker)
        hist = fetch_history(ticker, period)
        company = fetch_company_data(ticker)
        fin = get_financial_summary(company)
        metrics = calculate_market_metrics(hist)
        rows.append({
            "Company": name,
            "Ticker": ticker,
            "Latest price (₹)": metrics.get("latest_close"),
            "Period return (%)": metrics.get("period_return") * 100 if metrics.get("period_return") is not None else None,
            "Annualized volatility (%)": metrics.get("annualized_volatility") * 100 if metrics.get("annualized_volatility") is not None else None,
            "Max drawdown (%)": metrics.get("max_drawdown") * 100 if metrics.get("max_drawdown") is not None else None,
            "Revenue": fin.get("revenue"),
            "Profit margin (%)": fin.get("profit_margin") * 100 if fin.get("profit_margin") is not None else None,
            "Debt/equity (Yahoo)": fin.get("debt_to_equity"),
            "P/E": fin.get("trailing_pe"),
        })
        time.sleep(0.1)
    return pd.DataFrame(rows)


def show_metric(label, value, help_text=None):
    st.metric(label, value if value is not None else "N/A", help=help_text)


st.title("📊 AI Financial Intelligence Platform")
st.caption("Indian listed companies • Financial data + FinBERT news sentiment + explainable risk indicators")
with st.sidebar:
    st.header("Analysis settings")
    company_name = st.selectbox("Primary company", list(COMPANIES.keys()), index=0)
    ticker = COMPANIES[company_name]["ticker"]
    period_label = st.selectbox("Price history", ["6 months", "1 year", "2 years", "5 years"], index=1)
    period_map = {"6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y"}
    period = period_map[period_label]
    max_news = st.slider("News articles to analyze", min_value=5, max_value=30, value=15, step=5)
    run_news = st.checkbox("Run FinBERT sentiment analysis", value=True)
    st.divider()
    st.caption("Data providers")
    st.markdown("- Market/financial data: Yahoo Finance via `yfinance`")
    st.markdown("- News: NewsAPI if configured; otherwise Google News RSS")
    st.markdown("- NLP: ProsusAI/FinBERT")
    st.caption("Research/educational prototype. Not investment advice.")

tabs = st.tabs(["Company Overview", "News Intelligence", "Risk Analysis", "Company Comparison"])

with st.spinner(f"Loading {company_name} ({ticker}) data..."):
    history = fetch_history(ticker, period)
    company_data = fetch_company_data(ticker)
    financials = get_financial_summary(company_data)
    market_metrics = calculate_market_metrics(history)

news = pd.DataFrame()
scored_news = pd.DataFrame()
news_error = None
if run_news:
    with st.spinner("Fetching news and running FinBERT..."):
        try:
            news = fetch_news(company_name, ticker, max_news)
            if not news.empty:
                scored_news = analyze_news(news)
        except Exception as exc:
            news_error = str(exc)

risk_score, risk_components, risk_note = calculate_risk(financials, market_metrics, scored_news)

with tabs[0]:
    st.subheader(financials.get("long_name") or company_name)
    st.caption(f"{ticker} • {financials.get('sector') or COMPANIES[company_name]['sector']} • Source data may be delayed or incomplete.")
    if history.empty:
        st.error("No price history returned for this ticker. Check internet access or try another company.")
    else:
        last = market_metrics.get("latest_close")
        period_return = market_metrics.get("period_return")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_metric("Latest available close", f"₹{last:,.2f}" if last is not None else None)
        with c2:
            show_metric(f"{period_label} return", f"{period_return*100:+.2f}%" if period_return is not None else None)
        with c3:
            market_cap_label = "Estimated market cap" if financials.get("market_cap") is not None and company_data.get("info", {}).get("_marketCapEstimated") else "Market cap"
            show_metric(market_cap_label, fmt_currency(financials.get("market_cap")))
        with c4:
            show_metric("Risk score (0–100)", f"{risk_score:.1f}" if risk_score is not None else None, "Heuristic score; higher means more observed risk.")
        price_fig = px.line(history, x="Date", y="Close", title=f"{company_name}: adjusted historical closing price")
        price_fig.update_layout(xaxis_title="", yaxis_title="Price (₹)", hovermode="x unified")
        st.plotly_chart(price_fig, use_container_width=True)
        if "Volume" in history:
            volume_fig = px.bar(history.tail(90), x="Date", y="Volume", title="Daily trading volume (latest 90 sessions)")
            st.plotly_chart(volume_fig, use_container_width=True)
    st.markdown("#### Financial snapshot")
    rows = [
        ("Revenue (latest available)", fmt_currency(financials.get("revenue"))),
        ("Net income (latest available)", fmt_currency(financials.get("net_income"))),
        ("Total debt", fmt_currency(financials.get("total_debt"))),
        ("Cash and equivalents", fmt_currency(financials.get("total_cash"))),
        ("Trailing P/E", add_comma(financials.get("trailing_pe"))),
        ("Price-to-book", add_comma(financials.get("price_to_book"))),
        ("Profit margin", f"{financials['profit_margin']*100:.2f}%" if financials.get("profit_margin") is not None else "N/A"),
        ("Operating margin", f"{financials['operating_margin']*100:.2f}%" if financials.get("operating_margin") is not None else "N/A"),
        ("Return on equity", f"{financials['return_on_equity']*100:.2f}%" if financials.get("return_on_equity") is not None else "N/A"),
        ("Current ratio", add_comma(financials.get("current_ratio"))),
        ("Debt/equity (Yahoo field)", add_comma(financials.get("debt_to_equity"))),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]), use_container_width=True, hide_index=True)
    st.caption("Financial statement values are the latest values returned by the data provider, not necessarily the latest quarter. Verify material figures against company filings.")
    if financials.get("summary"):
        with st.expander("Company description from provider"):
            st.write(financials["summary"])

with tabs[1]:
    st.subheader("Financial News Intelligence")
    st.write("FinBERT classifies article text as positive, neutral, or negative. Sentiment is about wording in the available headline/description; it is not proof of future price direction.")
    if not run_news:
        st.info("Enable 'Run FinBERT sentiment analysis' in the sidebar to fetch and analyze news.")
    elif news_error:
        st.error(f"News/sentiment pipeline error: {news_error}")
    elif scored_news.empty:
        st.warning("No news articles were returned. Try again later or configure NEWSAPI_KEY.")
    else:
        counts = scored_news["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        avg_score = float(scored_news["sentiment_score"].mean())
        neg_share = float((scored_news["sentiment"] == "negative").mean())
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            show_metric("Articles analyzed", str(len(scored_news)))
        with n2:
            show_metric("Average sentiment", f"{avg_score:+.3f}", "Mean of positive probability minus negative probability.")
        with n3:
            show_metric("Negative headlines", f"{neg_share*100:.1f}%")
        with n4:
            show_metric("Mean model confidence", f"{scored_news['confidence'].mean()*100:.1f}%")
        sentiment_fig = px.bar(
            x=counts.index, y=counts.values, color=counts.index,
            labels={"x": "Sentiment", "y": "Article count"},
            title="Article sentiment distribution",
        )
        sentiment_fig.update_layout(showlegend=False)
        st.plotly_chart(sentiment_fig, use_container_width=True)
        display = scored_news[["published_at", "source", "title", "sentiment", "confidence", "sentiment_score", "url"]].copy()
        display["confidence"] = (display["confidence"] * 100).round(1).astype(str) + "%"
        display["sentiment_score"] = display["sentiment_score"].round(3)
        st.dataframe(display, use_container_width=True, hide_index=True, column_config={
            "url": st.column_config.LinkColumn("Article link", display_text="Open article"),
            "title": st.column_config.TextColumn("Headline", width="large"),
        })
        csv = scored_news.to_csv(index=False).encode("utf-8")
        st.download_button("Download analyzed news CSV", csv, file_name=f"{ticker.replace('.', '_')}_news_sentiment.csv", mime="text/csv")
        st.caption("Use publisher links to read original reporting. News sources, availability, licensing, and publication delays vary.")

with tabs[2]:
    st.subheader("Risk Analysis")
    if risk_score is None:
        st.warning("Insufficient data to calculate a score. Some providers do not return all financial fields.")
    else:
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Composite risk score", f"{risk_score:.1f}/100", help="A higher score means higher risk under this prototype's heuristic.")
            st.write(f"**Interpretation:** {risk_label(risk_score)}")
            st.caption(risk_note)
        with r2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={"text": "Heuristic risk score"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"thickness": 0.25},
                       "steps": [{"range": [0, 30]}, {"range": [30, 60]}, {"range": [60, 100]}]},
            ))
            gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)
    if risk_components:
        comp_df = pd.DataFrame([{"Risk factor": k, "Score (0–100)": v} for k, v in risk_components.items() if v is not None])
        st.markdown("#### Component scores")
        comp_fig = px.bar(comp_df, x="Risk factor", y="Score (0–100)", range_y=[0, 100], title="Higher component score = more risk under current rules")
        st.plotly_chart(comp_fig, use_container_width=True)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.markdown("#### Market risk metrics")
    if market_metrics:
        metrics_df = pd.DataFrame([
            {"Metric": "Annualized historical volatility", "Value": f"{market_metrics['annualized_volatility']*100:.2f}%" if market_metrics.get("annualized_volatility") is not None else "N/A", "Meaning": "Standard deviation of daily returns scaled by √252."},
            {"Metric": "Maximum drawdown", "Value": f"{market_metrics['max_drawdown']*100:.2f}%" if market_metrics.get("max_drawdown") is not None else "N/A", "Meaning": "Largest peak-to-trough fall in the selected historical window."},
            {"Metric": "Approx. 1-month return", "Value": f"{market_metrics['approx_1m_return']*100:+.2f}%" if market_metrics.get("approx_1m_return") is not None else "N/A", "Meaning": "Approximate last 21-session price change."},
            {"Metric": "Period return", "Value": f"{market_metrics['period_return']*100:+.2f}%" if market_metrics.get("period_return") is not None else "N/A", "Meaning": f"Return across selected window ({period_label})."},
            {"Metric": "Sharpe-like ratio (risk-free rate = 0)", "Value": f"{market_metrics['sharpe_rf0']:.2f}" if market_metrics.get("sharpe_rf0") is not None else "N/A", "Meaning": "Historical return-to-volatility proxy; not a forecast."},
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        dd_series = market_metrics.get("drawdown_series")
        if dd_series is not None and len(dd_series):
            dd_df = pd.DataFrame({"Date": history.loc[dd_series.index, "Date"].values, "Drawdown": dd_series.values})
            dd_fig = px.area(dd_df, x="Date", y="Drawdown", title="Historical drawdown")
            dd_fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(dd_fig, use_container_width=True)
    st.warning("Risk score is a transparent educational heuristic, not a statistically validated rating. Missing factors are omitted and remaining weights are re-normalized. Do not use it alone for investment decisions.")

with tabs[3]:
    st.subheader("Compare Indian Listed Companies")
    default_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    choices = st.multiselect(
        "Choose companies to compare",
        options=[v["ticker"] for v in COMPANIES.values()],
        default=default_tickers,
        format_func=lambda t: f"{TICKER_TO_NAME.get(t, t)} ({t})",
    )
    compare_period_label = st.selectbox("Comparison period", ["6 months", "1 year", "2 years", "5 years"], index=1, key="compare_period")
    compare_period = {"6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y"}[compare_period_label]
    if len(choices) < 2:
        st.info("Select at least two companies.")
    elif st.button("Run company comparison", type="primary"):
        with st.spinner("Fetching comparison data; this can take a little time..."):
            comparison = company_comparison(choices, compare_period)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        valid = comparison.dropna(subset=["Period return (%)"])
        if not valid.empty:
            fig = px.bar(valid, x="Company", y="Period return (%)", title=f"Price return comparison — {compare_period_label}")
            st.plotly_chart(fig, use_container_width=True)
        valid_vol = comparison.dropna(subset=["Annualized volatility (%)"])
        if not valid_vol.empty:
            fig2 = px.bar(valid_vol, x="Company", y="Annualized volatility (%)", title="Annualized historical volatility comparison")
            st.plotly_chart(fig2, use_container_width=True)
        st.download_button(
            "Download comparison CSV",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="company_comparison.csv",
            mime="text/csv",
        )
    else:
        st.caption("Select companies, then click Run company comparison. Some financial fields may be unavailable for certain tickers.")

st.divider()
st.caption(f"Prototype build • Last rendered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Data availability depends on upstream providers.")
