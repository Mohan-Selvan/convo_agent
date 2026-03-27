from __future__ import annotations

from typing import Any

import yfinance as yf
from langchain_core.tools import BaseTool, tool


@tool
def get_stock_price(symbol: str) -> dict[str, Any]:
    """Get live stock quote information for a ticker symbol like AAPL or MSFT."""
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info or {}

    current_price = info.get("last_price")
    currency = info.get("currency")
    day_high = info.get("day_high")
    day_low = info.get("day_low")

    if current_price is None:
        history = ticker.history(period="1d")
        if history.empty:
            raise RuntimeError(f"No price data found for symbol '{symbol}'.")
        current_price = float(history["Close"].iloc[-1])

    return {
        "price": current_price,
        "currency": currency,
        "day_high": day_high,
        "day_low": day_low,
    }


@tool
def get_company_profile(symbol: str) -> dict[str, Any]:
    """Get company profile details for a ticker symbol."""
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}

    summary = info.get("longBusinessSummary")
    if isinstance(summary, str) and len(summary) > 500:
        summary = summary[:500].rstrip() + "..."

    return {
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "summary": summary,
    }


@tool
def get_market_news(symbol: str) -> dict[str, Any]:
    """Get recent market news headlines for a ticker symbol."""
    ticker = yf.Ticker(symbol)
    news_items = ticker.news or []

    top = []
    for item in news_items[:3]:
        content = item.get("content") or {}
        top.append(
            {
                "title": content.get("title"),
                "publisher": content.get("provider"),
                "url": content.get("canonicalUrl", {}).get("url"),
            }
        )

    return {"items": top}


def get_tools() -> list[BaseTool]:
    return [get_stock_price, get_company_profile, get_market_news]
