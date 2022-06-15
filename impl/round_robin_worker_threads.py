from queue import Queue
import threading


class WorkerThread:
    def __init__(self, handler_cls, handler_init_params=(), worker_id=None, max_queue=16):
        self.handler_cls = handler_cls
        self.handler_init_params = handler_init_params
        self.worker_id = worker_id
        self.max_queue = max_queue

    def start(self):
        self.task_queue = Queue(self.max_queue)
        self.thread = threading.Thread(target=self._worker_entry)
        self.thread.start()

    def stop(self):
        if not hasattr(self, 'thread'):
            return
        self.task_queue.put(None)
        self.thread.join()
        del self.thread
        del self.task_queue

    def put(self, *args, **kwargs):
        self.task_queue.put((args, kwargs))

    def join(self):
        self.task_queue.join()

    def _worker_entry(self):
        handler = self.handler_cls(*self.handler_init_params)
        handler.set_worker_id(self.worker_id)
        with handler:
            while True:
                job = self.task_queue.get()
                if job is None:
                    self.task_queue.task_done()
                    break
                args, kwargs = job
                handler(*args, **kwargs)
                self.task_queue.task_done()


class RoundRobinWorkerThreads:
    def __init__(self, num_threads, handler_cls, handler_init_params=(), max_queue=16):
        self.num_threads = num_threads
        self.threads = [WorkerThread(handler_cls, handler_init_params, i, max_queue) for i in range(num_threads)]
        self.index = 0

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()

    def _start(self):
        for thread in self.threads:
            thread.start()

    def put(self, *args, **kwargs):
        self.threads[self.index].put(*args, **kwargs)
        self.index = (self.index + 1) % self.num_threads

    def join(self):
        for thread in self.threads:
            thread.join()

    def _stop(self):
        for thread in self.threads:
            thread.stop()


class BaseWorkerThreadHandler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_worker_id(self, id):
        self.worker_id = id

    def __call__(self, *args, **kwargs):
        pass
