# Analista Financiero Autónomo Multi-Agente

Proyecto final del curso **Text & Web Analytics**.  
El sistema implementa un analista financiero autónomo que, a partir de una consulta en lenguaje natural, coordina varios agentes para descargar datos de mercado, obtener noticias, aplicar NLP y generar un informe de análisis utilizando un modelo de lenguaje grande (LLM) ejecutado con Ollama.

---

## 1. Problema

En los mercados financieros actuales, el analista debe integrar información numérica (precios, retornos, indicadores) con información textual (noticias, reportes, opiniones). Hacerlo de forma manual es costoso y propenso a sesgos.

Este proyecto explora cómo una **arquitectura multi-agente** apoyada en **LLMs** puede asistir en esta tarea para consultas del tipo:

> “Analiza las acciones de AAPL y NVDA esta semana”.

---

## 2. Arquitectura y patrón de diseño

![Arquitectura multi-agente](Analista-financiero-multiagente\diagrams\Arquitectura_multiagente.png)

La solución sigue un **patrón de orquestación centralizada**:

- **CoordinatorAgent**  
  Orquesta el flujo completo, extrae los *tickers* desde la consulta y mantiene una memoria de interacciones.
- **MarketDataAgent**  
  Obtiene precios históricos desde Yahoo Finance usando `yfinance` y calcula variaciones porcentuales.
- **WebNewsAgent**  
  Recupera titulares de noticias para cada *ticker* mediante `requests + BeautifulSoup` y aplica un mecanismo de *fallback* cuando el *scraping* falla.
- **NLPAgent**  
  Limpia los textos, aplica TF–IDF y clasifica el sentimiento (positivo / negativo / neutral) con un modelo Naive Bayes.
- **LLMAnalystAgent**  
  Construye un *prompt* con precios, noticias y sentimiento, y solicita al LLM (llama3 vía Ollama) que genere un informe de análisis bursátil.

### Justificación del patrón elegido

Se eligió un patrón de **coordinador central + agentes especializados** por las siguientes razones:

- Permite mantener un **estado global** de la interacción (memoria de consultas).
- Hace más sencilla la **trazabilidad y depuración** del flujo durante el desarrollo del proyecto.
- Facilita el **desarrollo incremental**: cada agente se puede probar aislado (datos de mercado, noticias, NLP, LLM).
- La arquitectura se puede extender añadiendo nuevos agentes (por ejemplo, métricas de riesgo o indicadores técnicos) sin modificar los existentes.

No se optó por patrones más descentralizados (p.ej. agentes que se coordinen entre sí sin un coordinador explícito) porque, para el alcance del curso, el patrón centralizado es más fácil de implementar, explicar y evaluar.

---

## 3. Tecnologías usadas

- **Lenguaje:** Python 3.x
- **Entorno:** Google Colab / entorno local
- **Datos de mercado:** [`yfinance`](https://pypi.org/project/yfinance/)
- **Web scraping:** `requests`, `beautifulsoup4`
- **NLP clásico:** `nltk`, `scikit-learn` (TF–IDF + Multinomial Naive Bayes)
- **LLM local:** [Ollama](https://ollama.com/) + modelo `llama3`
- **Visualización (opcional):** `matplotlib`

---

## 4. Instalación

### 4.1. Clonar el repositorio

```bash
git clone https://github.com/USUARIO/analista-financiero-multiagente.git
cd analista-financiero-multiagente

