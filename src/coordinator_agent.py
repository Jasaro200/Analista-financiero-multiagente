"""
Agente coordinador del sistema multi-agente de análisis financiero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

import pandas as pd

from data_utils import get_market_data, summarize_market_data, get_news_for_tickers
from market_data_agent import MarketDataAgent
from web_news_agent import WebNewsAgent
from nlp_agent import NLPAgent
from llm_analyst_agent import LLMAnalystAgent


# --------- Utilidad: extraer tickers de una consulta --------- #

TICKER_PATTERN = r"\b[A-Z]{1,5}\b"


def extract_tickers_from_query(query: str) -> List[str]:
    """
    Extrae posibles tickers (códigos en mayúsculas) de la consulta del usuario.
    Ejemplo: 'Analiza AAPL y NVDA esta semana' -> ['AAPL', 'NVDA']
    """
    import re

    tickers = re.findall(TICKER_PATTERN, query)
    # Eliminar duplicados conservando orden
    seen = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# --------- CoordinatorAgent --------- #

DEFAULT_DAYS = 7


@dataclass
class CoordinatorAgent:
    """
    Agente coordinador: orquesta todo el flujo multi-agente
    y mantiene una memoria básica de las interacciones.
    """

    days: int = DEFAULT_DAYS
    interval: str = "1d"
    max_articles: int = 5
    llm_model_name: str = "llama3"

    # memoria simple de interacciones
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Inicializar agentes especializados
        self.market_agent = MarketDataAgent(days=self.days, interval=self.interval)
        self.web_agent = WebNewsAgent(max_articles=self.max_articles)
        self.nlp_agent = NLPAgent()
        self.llm_agent = LLMAnalystAgent(model_name=self.llm_model_name)

    def run(self, user_query: str, tickers: List[str] | None = None) -> Dict[str, Any]:
        """
        Ejecuta el flujo completo para una consulta del usuario.

        Si `tickers` es None, intenta extraerlos automáticamente
        desde la consulta.
        """
        if not tickers:
            tickers = extract_tickers_from_query(user_query)

        # 1. Datos de mercado
        market_raw = self.market_agent.get_data(tickers)
        market_summary = self.market_agent.summarize(market_raw)

        # 2. Noticias (web scraping + fallback)
        news_dict = self.web_agent.get_news(tickers)

        # 3. NLP / sentimiento
        sentiments = self.nlp_agent.analyze(news_dict)

        # 4. Informe del LLM
        llm_answer = self.llm_agent.build_and_call(
            user_query=user_query,
            market_summary=market_summary,
            sentiments=sentiments,
            news=news_dict,
        )

        # 5. Guardar en memoria
        record = {
            "user_query": user_query,
            "tickers": tickers,
            "market_raw": market_raw,
            "market_summary": market_summary,
            "news": news_dict,
            "sentiments": sentiments,
            "llm_answer": llm_answer,
        }
        self.history.append(record)

        # 6. Devolver resumen
        return record
