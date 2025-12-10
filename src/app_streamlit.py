import streamlit as st
import pandas as pd
from coordinator_agent import CoordinatorAgent

# Config básica de la página
st.set_page_config(
    page_title="Analista Financiero Autónomo",
    layout="wide",
)

st.title("📊 Analista Financiero Autónomo")
st.write(
    "Demo del proyecto de Arquitecturas Agénticas para Text & Web Analytics. "
    "Las recomendaciones son **simuladas** y con fines académicos."
)

# --- Sidebar con parámetros ---
st.sidebar.header("Parámetros de análisis")
default_query = "Analiza las acciones de AAPL y NVDA esta semana"

user_query = st.text_input(
    "Consulta en lenguaje natural:",
    value=default_query,
    help="Ejemplo: Analiza EC esta semana",
)

days = st.sidebar.slider("Días a analizar", min_value=3, max_value=30, value=7)
max_articles = st.sidebar.slider(
    "Máx. noticias por ticker", min_value=1, max_value=10, value=3
)

llm_model = st.sidebar.text_input("Modelo LLM (Ollama)", value="llama3")

# Crear el coordinador una vez por ejecución
coordinator = CoordinatorAgent(
    days=days,
    interval="1d",
    max_articles=max_articles,
    llm_model_name=llm_model,
)

if st.button("▶ Ejecutar análisis"):
    if not user_query.strip():
        st.warning("Por favor escribe una consulta.")
    else:
        with st.spinner("Ejecutando agentes y generando informe..."):
            result = coordinator.run(user_query)

        # --- Sección 1: Tickers detectados ---
        st.subheader("Tickers detectados")
        st.write(result["tickers"])

        # --- Sección 2: Resumen de mercado ---
        st.subheader("Resumen de mercado (últimos días)")
        st.dataframe(result["market_summary"])

                # --- Sección 3: Gráfico de precios ---
        st.subheader("Evolución de precios")

        precios_df = pd.DataFrame()

        for t, df in result["market_raw"].items():
            # Asumimos que data_utils devuelve una columna 'Date'
            # Si no, cambia 'Date' por el nombre real de la columna de fecha
            if "Date" in df.columns:
                serie = df.set_index("Date")["Close"]
            else:
                # Si la fecha viene en el índice, usamos el índice tal cual
                serie = df["Close"]
                serie.index = df.index  # por claridad

            serie.name = t            # nombre de la serie = ticker
            precios_df[t] = serie     # cada columna = un ticker

        if not precios_df.empty:
            # Streamlit usa el índice como eje X y cada columna como una línea
            st.line_chart(precios_df)
            # si prefieres área:
            # st.area_chart(precios_df)
        else:
            st.info("No se encontraron datos de precios para graficar.")

        # --- Sección 4: Sentimiento por ticker ---
        st.subheader("Sentimiento por ticker (Naive Bayes)")
        for t, s in result["sentiments"].items():
            st.markdown(
                f"**{t}** → global: `{s['sentiment_global']}` "
                f"(pos: {s['num_pos']}, neg: {s['num_neg']}, neu: {s['num_neu']})"
            )

        # --- Sección 5: Algunas noticias ---
        st.subheader("Titulares usados en el análisis")
        for t, info in result["news"].items():
            st.markdown(f"**{t}**")
            for art in info["raw"]:
                st.write("•", art["titulo"])

        # --- Sección 6: Informe del analista (LLM) ---
        st.subheader("Informe del Analista (LLM / Ollama)")
        st.markdown(result["llm_answer"])
