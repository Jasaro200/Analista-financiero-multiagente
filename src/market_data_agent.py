"""
Agente de datos de mercado.

Envuelve las funciones de data_utils para descargar y resumir
precios históricos desde Yahoo Finance.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from data_utils import get_market_data, summarize_market_data


class MarketDataAgent:
    """
    Agente responsable de obtener y resumir datos de mercado.
    """

    def __init__(self, days: int = 7, interval: str = "1d") -> None:
        self.days = days
        self.interval = interval

    def get_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Descarga datos brutos de mercado para los tickers indicados.
        """
        return get_market_data(tickers, days=self.days, interval=self.interval)

    def summarize(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Devuelve un resumen con variación porcentual por ticker.
        """
        if not data_dict:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "fecha_inicio",
                    "fecha_fin",
                    "precio_inicio",
                    "precio_fin",
                    "variacion_pct",
                ]
            )
        return summarize_market_data(data_dict)
