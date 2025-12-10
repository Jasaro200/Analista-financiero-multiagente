"""
Agente de NLP: limpieza de texto y análisis de sentimiento con TF-IDF + Naive Bayes.
"""

from __future__ import annotations

import re
from typing import Dict, List, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class NLPAgent:
    """
    Aplica un pipeline sencillo de NLP para estimar el sentimiento
    (positivo / negativo / neutral) de titulares de noticias.
    """

    def __init__(self) -> None:
        # Asegurar recursos mínimos de NLTK
        self._ensure_nltk_resources()

        self.spanish_stopwords = set(stopwords.words("spanish"))

        # Entrenar modelo Naive Bayes con un corpus pequeño de ejemplo
        self.vectorizer = TfidfVectorizer()
        self.clf = MultinomialNB()
        self.label_to_int = {"negativo": 0, "neutral": 1, "positivo": 2}
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

        self._train_model()

    # ------------------ Utilidades internas ------------------ #

    def _ensure_nltk_resources(self) -> None:
        """
        Intenta descargar los recursos necesarios de NLTK si no están presentes.
        """
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("tokenizers/punkt_tab/spanish")
        except LookupError:
            # En algunas versiones nuevas de NLTK se requiere este recurso
            try:
                nltk.download("punkt_tab")
            except Exception:
                pass

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

    def _clean_text(self, text: str) -> str:
        """
        Normaliza el texto: minúsculas, eliminación de símbolos y stopwords.
        """
        text = text.lower()
        text = re.sub(r"[^a-záéíóúñü0-9\s]", " ", text)
        tokens = word_tokenize(text, language="spanish")
        tokens = [
            t for t in tokens
            if t not in self.spanish_stopwords and len(t) > 2
        ]
        return " ".join(tokens)

    def _train_model(self) -> None:
        """
        Entrena el modelo Naive Bayes sobre un pequeño corpus manual.
        """
        textos_entrenamiento = [
            # Positivos
            "la compañía reporta resultados trimestrales mejor de lo esperado",
            "analistas revisan al alza sus perspectivas sobre la acción",
            "aumentan las ventas y el mercado reacciona positivamente",
            "fuerte demanda en el sector impulsa el precio de la acción",
            # Negativos
            "la empresa registra pérdidas mayores de lo previsto",
            "recorte de guía y preocupaciones sobre la demanda",
            "caída fuerte de la acción tras resultados decepcionantes",
            "el regulador investiga a la compañía por posibles irregularidades",
            # Neutrales
            "el mercado se mantiene estable sin cambios relevantes",
            "noticias mixtas en el sector y el precio se mantiene lateral",
            "la compañía celebra junta de accionistas sin anuncios relevantes",
            "los analistas esperan pocas sorpresas en el próximo trimestre",
        ]

        etiquetas = [
            "positivo",
            "positivo",
            "positivo",
            "positivo",
            "negativo",
            "negativo",
            "negativo",
            "negativo",
            "neutral",
            "neutral",
            "neutral",
            "neutral",
        ]

        textos_limpios = [self._clean_text(t) for t in textos_entrenamiento]
        X = self.vectorizer.fit_transform(textos_limpios)
        y = np.array([self.label_to_int[e] for e in etiquetas])
        self.clf.fit(X, y)

    # ------------------ API pública ------------------ #

    def analyze(self, news_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Recibe el diccionario de noticias por ticker:

          {
            "AAPL": { "raw": [{"titulo": "...", "url": "..."}, ...], ... },
            ...
          }

        y devuelve, para cada ticker, un resumen de sentimiento:

          {
            "AAPL": {
              "sentiments": ["positivo", "negativo", ...],
              "sentiment_global": "positivo",
              "num_pos": 2,
              "num_neg": 1,
              "num_neu": 0
            },
            ...
          }
        """
        result: Dict[str, Dict[str, Any]] = {}

        for ticker, info in news_dict.items():
            raw_articles = info.get("raw", [])
            titles = [a.get("titulo", "") for a in raw_articles if a.get("titulo")]

            if not titles:
                # Si no hay títulos, devolvemos todo como neutral.
                result[ticker] = {
                    "sentiments": [],
                    "sentiment_global": "neutral",
                    "num_pos": 0,
                    "num_neg": 0,
                    "num_neu": 0,
                }
                continue

            cleaned_titles = [self._clean_text(t) for t in titles]
            X = self.vectorizer.transform(cleaned_titles)
            y_pred_int = self.clf.predict(X)
            y_pred_labels = [self.int_to_label[i] for i in y_pred_int]

            num_pos = sum(1 for s in y_pred_labels if s == "positivo")
            num_neg = sum(1 for s in y_pred_labels if s == "negativo")
            num_neu = sum(1 for s in y_pred_labels if s == "neutral")

            # Regla de mayoría simple
            counts = {
                "positivo": num_pos,
                "negativo": num_neg,
                "neutral": num_neu,
            }
            sentiment_global = max(counts, key=counts.get)

            result[ticker] = {
                "sentiments": y_pred_labels,
                "sentiment_global": sentiment_global,
                "num_pos": num_pos,
                "num_neg": num_neg,
                "num_neu": num_neu,
            }

        return result
