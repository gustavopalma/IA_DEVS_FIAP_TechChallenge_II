
from google import genai
from src.monitoramento import (
    iniciar_monitoramento,
    finalizar_monitoramento
)


def explicar_diagnostico(dados, previsao, probabilidade, modelo_llm="gemini-3.8-flash"):
    prompt = f"""
    Você é um assistente de apoio à interpretação de resultados
    de machine learning na área da saúde.

    Contexto: classificação de risco de Síndrome dos Ovários
    Policísticos (SOP).

    Dados clínicos:
    {dados}

    Nesse caso considere para o ciclo regular o valor 2 e para ciclo irregular o valor 4

    Previsão do classificador:
    {"Positivo para SOP" if previsao == 1 else "Negativo para SOP"}

    Probabilidade estimada da classe positiva:
    {probabilidade:.2%}

    Explique em português brasileiro:
    1. O significado da previsão.
    2. O que os dados apresentados representam.
    3. As limitações da previsão e da probabilidade estimada.
    4. Quais informações adicionais poderiam ser úteis
       para uma avaliação clínica.

    Regras:
    - Não apresente a previsão como diagnóstico confirmado.
    - Não invente sintomas, exames ou informações.
    - Não afirme que uma variável causou a previsão sem
      receber dados de explicabilidade do modelo.
    - Não recomende tratamentos ou medicamentos.
    - Deixe claro que a decisão clínica cabe ao profissional.
    """


    inicio = iniciar_monitoramento()
    modelo_utilizado = modelo_llm

    try:
        with genai.Client() as cliente:
            resposta = cliente.models.generate_content(
                model=modelo_utilizado,
                contents=prompt
            )

        if not resposta.text:
            raise ValueError("O Gemini retornou uma resposta vazia")

        finalizar_monitoramento(
            "gemini",
            inicio,
            sucesso=True,
            modelo=modelo_utilizado
        )

        return resposta.text

    except Exception as erro:
        finalizar_monitoramento(
            "gemini",
            inicio,
            sucesso=False,
            modelo=modelo_utilizado,
            erro=type(erro).__name__,
            codigo=getattr(erro, "code", "N/A")
        )

        raise
