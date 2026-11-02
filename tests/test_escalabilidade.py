
import threading
import time

from src.escalabilidade import Escalabilidade


def test_escalabilidade_automatica():
    escalabilidade = Escalabilidade(
        minimo=1,
        maximo=3,
        tarefas_por_worker=1
    )

    concluidas = []
    lock = threading.Lock()

    def tarefa(numero):
        time.sleep(0.1)

        with lock:
            concluidas.append(numero)

    try:
        for i in range(12):
            escalabilidade.enviar(lambda i=i: tarefa(i))

        escalabilidade.aguardar()

        assert len(concluidas) == 12
        assert sorted(concluidas) == list(range(12))

        assert max(
            registro["workers"]
            for registro in escalabilidade.historico
        ) <= 3

        assert max(
            registro["workers"]
            for registro in escalabilidade.historico
        ) > 1

        assert escalabilidade.fila.empty()

    finally:
        escalabilidade.encerrar()
