import logging
import time
from pathlib import Path

# Diretório de logs na raiz do projeto
DIRETORIO_LOGS = Path(__file__).resolve().parent.parent / "logs"
DIRETORIO_LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tech_challenge")
logger.setLevel(logging.INFO)

if not logger.handlers:
    arquivo = logging.FileHandler(
        DIRETORIO_LOGS / "aplicacao.log",
        encoding="utf-8"
    )

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    arquivo.setFormatter(formato)
    logger.addHandler(arquivo)


def iniciar_monitoramento():
    return time.perf_counter()


def finalizar_monitoramento(operacao, inicio, sucesso=True, **metricas):
    duracao = time.perf_counter() - inicio

    detalhes = " | ".join(
        f"{chave}={valor}" for chave, valor in metricas.items()
    )

    mensagem = (
        f"operacao={operacao} | "
        f"duracao={duracao:.4f}s | "
        f"sucesso={sucesso}"
    )

    if detalhes:
        mensagem += f" | {detalhes}"

    if sucesso:
        logger.info(mensagem)
    else:
        logger.error(mensagem)

    return duracao
