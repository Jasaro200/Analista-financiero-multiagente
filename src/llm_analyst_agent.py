"""
Agente analista que usa un LLM (llama3 vía Ollama) para generar el informe final.
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd
import ollama


class LLMAnalystAgent:
    """
    Construye un prompt con los datos de mercado, sentimiento y noticias,
    y llama al modelo de lenguaje para generar un informe textual.
    """

    def __init__(self, model_name: str = "llama3") -> None:
        self.model_name = model_name

    # ------------------ API pública ------------------ #

    def build_and_call(
        self,
        user_query: str,
        market_summary: pd.DataFrame,
        sentiments: Dict[str, Dict[str, Any]],
        news: Dict[str, Dict[str, Any]],
    ) -> str:
        prompt = self._build_prompt(user_query, market_summary, sentiments, news)
        return self._call_llm(prompt)

    # ------------------ Internas ------------------ #

    def _build_prompt(
        self,
        user_query: str,
        market_summary: pd.DataFrame,
        sentiments: Dict[str, Dict[str, Any]],
        news: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Construye un prompt en español para el modelo.
        """
        summary_text = market_summary.to_string(index=False)

        sentiment_lines = []
        for t, s in sentiments.items():
            line = (
                f"{t}: global={s['sentiment_global']}, "
                f"pos={s['num_pos']}, neg={s['num_neg']}, neu={s['num_neu']}"
            )
            sentiment_lines.append(line)
        sentiment_text = "\n".join(sentiment_lines)

        news_text_blocks = []
        for t, info in news.items():
            titles = [a.get("titulo", "") for a in info.get("raw", []) if a.get("titulo")]
            joined = " | ".join(titles)
            news_text_blocks.append(f"{t}: {joined}")
        news_text = "\n".join(news_text_blocks)

        prompt = f"""
Eres un analista financiero que escribe informes cortos y claros en español.
Estás trabajando en un entorno académico: NO debes dar recomendaciones reales,
solo recomendaciones simuladas y siempre con un aviso de que no es asesoría financiera.

Consulta del usuario:
\"\"\"{user_query}\"\"\"

Resumen de mercado (últimos días):
{summary_text}

Resumen de sentimiento por ticker:
{sentiment_text}

Titulares recientes por ticker:
{news_text}

Tareas:

1. Describe brevemente el comportamiento reciente de cada acción.
2. Relaciona el movimiento de precios con el contexto de noticias y el sentimiento.
3. Propón una recomendación SIMULADA para cada acción (comprar, mantener, vender),
   explicando la lógica de forma sencilla.
4. Termina con una nota clara de que este análisis es solo con fines académicos
   y no constituye asesoría financiera.

Escribe el informe en formato de texto, usando subtítulos por ticker.
"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Llama al modelo de Ollama y devuelve solo el texto de respuesta.
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            # Fallback en caso de error con Ollama
            return (
                "No fue posible contactar al modelo LLM a través de Ollama.\n"
                f"Error: {e}\n\n"
                "Sin embargo, el sistema sí generó los datos cuantitativos y de sentimiento "
                "que pueden consultarse en las otras salidas."
            )
