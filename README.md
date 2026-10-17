# AI Financial Intelligence Platform

A student research prototype for Indian listed companies. It combines market and financial data, FinBERT-based news sentiment, explainable heuristic risk indicators, and a company-comparison dashboard.

## Features

- Company overview with historical prices, financial statement snapshots, valuation/profitability ratios, and price charts.
- News intelligence using FinBERT (`ProsusAI/finbert`) with positive/neutral/negative probabilities and links to sources.
- Risk analysis: annualized historical volatility, maximum drawdown, approximate one-month/period return, Sharpe-like ratio, debt, liquidity/profitability, and news sentiment.
- Company comparison across selected NSE-listed tickers.
- CSV exports for analyzed news and company comparisons.
- NewsAPI support when `NEWSAPI_KEY` is configured, with Google News RSS fallback.

## Data-source notes

- Market/financial data: Yahoo Finance via `yfinance`; NSE symbols usually use `.NS` (e.g. `RELIANCE.NS`, `TCS.NS`). Data may be delayed, incomplete, adjusted, or changed by the provider.
- News: NewsAPI if a key is supplied; otherwise Google News RSS is used to discover headlines and publisher links. NewsAPI's free Developer plan is for development/testing and has limitations/delays. Review provider terms before public/commercial use.
- FinBERT: `ProsusAI/finbert` from Hugging Face. First run downloads model files and can take several minutes. GPU is helpful but not mandatory.
- The app does not scrape full publisher article text. It analyzes available titles/descriptions.
- Financial fields from Yahoo Finance are not a substitute for audited filings. Verify key numbers against NSE/BSE/company investor-relations disclosures.
- This project is for education and research, not financial advice. Its 0–100 risk score is a transparent heuristic, not a validated credit/rating model.

## Project structure

```text
ai_financial_intelligence_platform/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Run in Google Colab

1. Upload this ZIP to Google Drive or your Colab session.
2. In a new Colab notebook, run:

```python
!pip -q install -r /content/ai_financial_intelligence_platform/requirements.txt
```

If you uploaded the ZIP to `/content`, unzip it first:

```python
!unzip -q /content/ai_financial_intelligence_platform.zip -d /content/
```

3. Start Streamlit:

```python
!streamlit run /content/ai_financial_intelligence_platform/app.py --server.port 8501 --server.headless true &>/content/streamlit.log &
```

4. For a temporary Colab preview, use a tunnel only if permitted by your institution/network. One option is:

```python
!npm -s install -g localtunnel
!lt --port 8501
```

Open the URL printed by localtunnel. Treat the URL as public to anyone who obtains it; do not put private information in this demo. If tunnel setup is blocked, run the app locally or deploy via Streamlit Community Cloud.

For Colab, make sure the path matches where you unzipped the project. The first FinBERT call may take a few minutes while the model downloads.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional NewsAPI key

Set an environment variable before launching:

```bash
# macOS/Linux
export NEWSAPI_KEY="your_key_here"
streamlit run app.py
```

Windows PowerShell:

```powershell
$env:NEWSAPI_KEY="your_key_here"
streamlit run app.py
```

You may also create `.streamlit/secrets.toml` locally:

```toml
NEWSAPI_KEY = "your_key_here"
```

Never commit `secrets.toml` or API keys to GitHub. If no key is configured, the app tries Google News RSS.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `.streamlit/config.toml`.
2. Open Streamlit Community Cloud and create an app from that repository.
3. Select `app.py` as the entry point.
4. If using NewsAPI, add `NEWSAPI_KEY = "your_key_here"` in the app's Secrets settings. Do not commit secrets.
5. Wait for dependency installation and model download. A first cold start can be slow. If memory is insufficient, use a smaller deployment or move FinBERT inference to a separate inference endpoint.
6. Test with at least two tickers and verify that prices, dates, links, and financial ratios make sense.

## Suggested academic validation

- Record the date/time and source for each fetched data point.
- Manually verify selected company figures against official disclosures.
- Test sentiment on manually labeled finance headlines and report accuracy, precision, recall, and a confusion matrix.
- Compare risk signals with subsequent realized volatility/drawdowns using a time-based split; do not randomly mix future and past data.
- Add unit tests for risk calculations and missing-data behavior before presenting the score as meaningful.

## Known limitations

- Yahoo Finance may omit or change some statement fields; the app falls back across common labels but may show `N/A`.
- The RSS fallback is a discovery feed; the number and age of results can vary.
- News relevance is imperfect: a company acronym or ticker may appear in unrelated stories.
- The score weights and thresholds are starter heuristics and need documented calibration for academic claims.
- This app does not make buy/sell recommendations or guarantee returns.
