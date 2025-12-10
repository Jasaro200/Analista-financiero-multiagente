"""
Funciones auxiliares para adquisición de datos de mercado y noticias.
"""

from __future__ import annotations

import re
import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


# --------- Datos de mercado (Yahoo Finance) --------- #

def get_market_data(
    tickers: List[str],
    days: int = 7,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Descarga datos de mercado recientes para una lista de tickers usando yfinance.

    Retorna un diccionario {ticker: DataFrame} con columnas típicas de OHLC.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=days)

    data_dict: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        print(f"Descargando datos para {t} ({start} -> {end})...")
        df = yf.download(
            t,
            start=start,
            end=end,
            interval=interval,
            progress=False,
        )
        if df.empty:
            print(f"⚠️  Sin datos para {t}")
            continue
        df.reset_index(inplace=True)
        data_dict[t] = df

    return data_dict


def summarize_market_data(
    data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Construye un DataFrame resumen con precio inicial, final y variación porcentual.
    """
    rows = []

    for t, df in data_dict.items():
        df_sorted = df.sort_values(by=df.columns[0])
        first_row = df_sorted.iloc[0]
        last_row = df_sorted.iloc[-1]

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        price_start = float(first_row[price_col])
        price_end = float(last_row[price_col])
        var_pct = (price_end - price_start) / price_start * 100

        rows.append(
            {
                "ticker": t,
                "fecha_inicio": first_row[df.columns[0]].date()
                if hasattr(first_row[df.columns[0]], "date")
                else first_row[df.columns[0]],
                "fecha_fin": last_row[df.columns[0]].date()
                if hasattr(last_row[df.columns[0]], "date")
                else last_row[df.columns[0]],
                "precio_inicio": round(price_start, 2),
                "precio_fin": round(price_end, 2),
                "variacion_pct": round(var_pct, 2),
            }
        )

    return pd.DataFrame(rows)


# --------- Noticias (web scraping + fallback) --------- #

def _fake_news_for_ticker(ticker: str, n: int = 3) -> List[Dict[str, Optional[str]]]:
    """
    Genera titulares sintéticos cuando el scraping falla.
    """
    base = [
        f"{ticker}: La compañía reporta resultados trimestrales mejor de lo esperado.",
        f"{ticker}: Analistas revisan sus perspectivas para la acción.",
        f"{ticker}: Noticias mixtas en el sector impactan el desempeño reciente.",
    ]

    return [{"titulo": t, "url": None} for t in base[:n]]


def get_news_for_tickers(
    tickers: List[str],
    max_articles: int = 5,
    limpiar: bool = False,
    limpiar_fn=None,
) -> Dict[str, Dict[str, List]]:
    """
    Intenta obtener titulares de noticias para cada ticker.
    Si Yahoo Finanzas falla (códigos 4xx/5xx), usa titulares sintéticos.

    Retorna:
      {
        "AAPL": {
            "raw": [{"titulo": "...", "url": "..."}, ...],
            "clean": ["titulo limpio 1", ...]  # solo si limpiar=True
        },
        ...
      }
    """
    base_url = "https://es-us.finanzas.yahoo.com/quote/{ticker}/news/"
    result: Dict[str, Dict[str, List]] = {}

    for t in tickers:
        url = base_url.format(ticker=t)
        print(f"Buscando noticias para {t} en {url}")

        articles: List[Dict[str, Optional[str]]] = []

        try:
            resp = requests.get(url, timeout=10)
            print(f"  -> status HTTP: {resp.status_code}")

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                links = soup.find_all("a", href=True)

                for a in links:
                    href = a["href"]
                    text = a.get_text(strip=True)
                    if "/news/" in href and text:
                        full_url = (
                            href if href.startswith("http") else "https://es-us.finanzas.yahoo.com" + href
                        )
                        articles.append({"titulo": text, "url": full_url})
                        if len(articles) >= max_articles:
                            break

            if not articles or resp.status_code != 200:
                print("⚠️ Respuesta no 200 o sin titulares. Se usará fallback.")
                articles = _fake_news_for_ticker(t, n=max_articles)

        except Exception as e:
            print(f"⚠️ Error al obtener noticias de {t}: {e}")
            articles = _fake_news_for_ticker(t, n=max_articles)

        # Limpieza opcional de los títulos
        if limpiar and limpiar_fn is not None:
            clean_titles = [limpiar_fn(a["titulo"]) for a in articles]
        else:
            clean_titles = None

        result[t] = {"raw": articles, "clean": clean_titles}

    return result
