"""
Agente de adquisición de noticias financieras (web scraping + fallback).
"""

from __future__ import annotations

from typing import Dict, List, Any

from data_utils import get_news_for_tickers


class WebNewsAgent:
    """
    Agente responsable de obtener titulares de noticias para cada ticker.
    """

    def __init__(self, max_articles: int = 5) -> None:
        self.max_articles = max_articles

    def get_news(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Llama a la función de utilería que intenta hacer scraping a Yahoo Finanzas
        y, en caso de error, usa titulares de ejemplo (fallback).

        No limpia los textos; la limpieza se delega al NLPAgent.
        """
        return get_news_for_tickers(
            tickers,
            max_articles=self.max_articles,
            limpiar=False,
            limpiar_fn=None,
        )
