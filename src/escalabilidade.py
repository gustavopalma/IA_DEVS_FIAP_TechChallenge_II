import logging
import math
import queue
import threading
import time

from src.monitoramento import logger


class Escalabilidade:
    def __init__(self, minimo=1, maximo=5, tarefas_por_worker=2):
        if minimo < 1 or maximo < minimo or tarefas_por_worker < 1:
            raise ValueError("Configuração de escalabilidade inválida")

        self.minimo = minimo
        self.maximo = maximo
        self.tarefas_por_worker = tarefas_por_worker

        self.fila = queue.Queue()
        self.workers = []
        self.ativos = 0
        self.desejados = minimo
        self.executando = True
        self.lock = threading.Lock()
        self.historico = []

        for _ in range(minimo):
            self._criar_worker()

        self.controlador = threading.Thread(
            target=self._controlar, daemon=True
        )
        self.controlador.start()

    def _criar_worker(self):
        thread = threading.Thread(
            target=self._processar, daemon=True
        )
        self.workers.append(thread)
        thread.start()

    def _processar(self):
        atual = threading.current_thread()

        while True:
            try:
                tarefa = self.fila.get(timeout=0.2)
            except queue.Empty:
                with self.lock:
                    if not self.executando or len(self.workers) > self.desejados:
                        self.workers.remove(atual)
                        return
                continue

            with self.lock:
                self.ativos += 1

            try:
                tarefa()
            except Exception as erro:
                logger.error(
                    "operacao=worker | sucesso=False | erro=%s",
                    type(erro).__name__
                )
            finally:
                with self.lock:
                    self.ativos -= 1
                self.fila.task_done()

    def _controlar(self):
        while self.executando:
            with self.lock:
                pendentes = self.fila.qsize()
                demanda = pendentes + self.ativos

                self.desejados = max(
                    self.minimo,
                    min(
                        self.maximo,
                        math.ceil(demanda / self.tarefas_por_worker)
                    )
                )

                existentes = len(self.workers)
                criar = max(0, self.desejados - existentes)

                self.historico.append({
                    "tempo": time.perf_counter(),
                    "pendentes": pendentes,
                    "ativos": self.ativos,
                    "workers": existentes,
                    "desejados": self.desejados
                })

            for _ in range(criar):
                with self.lock:
                    if len(self.workers) < self.desejados:
                        self._criar_worker()

            time.sleep(0.1)

    def enviar(self, tarefa):
        if not self.executando:
            raise RuntimeError("Escalabilidade encerrada")
        self.fila.put(tarefa)

    def aguardar(self):
        self.fila.join()

        # Permite observar a redução automática após a carga
        time.sleep(0.8)

    def encerrar(self):
        self.aguardar()
        self.executando = False
        self.controlador.join()

        for worker in list(self.workers):
            worker.join()
