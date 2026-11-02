
from unittest.mock import MagicMock, patch

from src.llm_gemini import explicar_diagnostico


def test_explicar_diagnostico():
    resposta = MagicMock()
    resposta.text = "Esta previsão não substitui uma avaliação médica."

    cliente = MagicMock()
    cliente.__enter__.return_value = cliente
    cliente.models.generate_content.return_value = resposta

    dados = {
        "Follicle No. (R)": 7,
        "Follicle No. (L)": 6,
        "BMI": 24.0
    }

    with patch("src.llm_gemini.genai.Client", return_value=cliente):
        texto = explicar_diagnostico(
            dados,
            previsao=0,
            probabilidade=0.15
        )

    assert texto == resposta.text
    cliente.models.generate_content.assert_called_once()

    argumentos = cliente.models.generate_content.call_args.kwargs

    assert "model" in argumentos
    assert "contents" in argumentos
    assert "BMI" in argumentos["contents"]
