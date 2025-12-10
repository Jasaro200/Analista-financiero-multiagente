"""
Punto de entrada del Analista Financiero Autónomo.

Este script:
- Crea una instancia de CoordinatorAgent
- Lanza una consulta de ejemplo en lenguaje natural
- Muestra por pantalla el resumen y el informe del LLM
"""

from coordinator_agent import CoordinatorAgent


def run_demo():
    """
    Ejecuta una demo simple del sistema multi-agente.
    """
    # Puedes cambiar esta frase para otras pruebas:
    user_query = "Analiza las acciones de AAPL y NVDA esta semana"

    # Crear el coordinador con parámetros por defecto
    coordinator = CoordinatorAgent(
        days=7,
        interval="1d",
        max_articles=3,
        llm_model_name="llama3",
    )

    # Ejecutar el flujo completo
    result = coordinator.run(user_query)

    # Mostrar resultados básicos en consola
    print("\n=== Consulta del usuario ===")
    print(user_query)

    print("\n=== Tickers detectados ===")
    print(coordinator.history[-1]["tickers"])

    print("\n=== Resumen de mercado (últimos días) ===")
    print(result["market_summary"])

    print("\n=== Sentimiento por ticker ===")
    for t, s in result["sentiments"].items():
        print(f"{t}: {s}")

    print("\n=== Informe del Analista (LLM/Ollama) ===\n")
    print(result["llm_answer"])


if __name__ == "__main__":
    run_demo()