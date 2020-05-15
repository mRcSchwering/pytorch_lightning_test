"""
Using the convenience of multiprocessing.Pool with pytorch.

Originally, I wanted to start several parallel pytorch training runs from one python process.
I can do that with os.fork() but multprocessing.Pool is much more convenient.
However, the processes from that are daemonized and cannot themself have children.
Fortunately, I found this solution:
https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
"""
import os
from contextlib import contextmanager
import multiprocessing
from multiprocessing import Value, Process, Manager
from src.config import N_GPUS


class NoDaemonProcess(multiprocessing.Process):
    """A Process that has 'daemon' attribute always return False"""
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(multiprocessing.pool.Pool):
    """
    Extend multiprocessing.pool.Pool with NonDaemonProcess.
    Cannot extend multiprocessing.Pool as it's just a wrapper.
    """
    Process = NoDaemonProcess

    def __init__(self):
        super().__init__(processes=max(1, N_GPUS))


def subprocess(target: callable, kwargs: dict) -> dict:
    """
    Run target function in another process. It must return a dict.

.   This can help making the execution of `target` save.
    E.g. when using optuna's `study.optimize(..., n_jobs=2)` with
    pytorch's `DataLoader(..., num_workers=4)`.
    """
    def process(gpus: list, _return_dict: dict, **kwargs):
        res = target(gpus=gpus, **kwargs)
        _return_dict.update(res)

    res = {}
    with Manager() as manager:
        return_dict = manager.dict()
        kwargs.update({'_return_dict': return_dict})
        proc = Process(target=process, kwargs=kwargs)
        proc.start()
        proc.join()
        res.update(return_dict)
    return res


class GpuQueue:
    """
    Queue for processes or threads to exchange which GPUs they are using.

    If you fork your main process/thread somehow into multiple workers
    (e.g. multiprocessing or -threading) use this queue to make your workers
    communicate among each other which GPU(s) they are currently using.

    Set `n_available_gpus` for the number of GPUs in the pool.
    Leave it `None` to infer the number of GPUs. It is safer to keep
    `n_available_gpus=None` and restrict the GPUs using `CUDA_VISIBLE_DEVICES`.

    Initialize in main process before the fork.
    Then pass a reference as argument to the workers.
    Within the worker use the context manager to get GPUs.
    This blocks these GPUs for other workers.
    When the context exits, the GPUs are released again.

    :Example:
        def process(queue: GpuQueue, config: dict):
            with queue.n_gpus() as gpus:
                time.sleep(1)
                return f'pid:{os.getpid()} used GPUs {gpus}'
        
        if __name__ == '__main__':
            inputs = ['set1', 'set2', 'set3', 'set4']
            gq = GpuQueue()
            
            with Pool(processes=2) as pool:
                results = pool.starmap(process, [(gq, d) for d in inputs])
    
    Note: Be careful with multithreading and multiprocessing issues when using
    together with pytorch. pytorch's `Dataloader` for example also use multiprocessing.
    This can lead to weird bugs (see `subprocess` above).
    """

    def __init__(self, n_available_gpus: int = None):
        self.queue = multiprocessing.Manager().Queue()
        if n_available_gpus is None:
            n_available_gpus = N_GPUS
        self.n_available_gpus = n_available_gpus
        if self.n_available_gpus > 0:
            for idx in range(self.n_available_gpus):
                self.queue.put(idx)


    @contextmanager
    def n_gpus(self, n: int = 1) -> list:
        """
        Context manager to get and block ids of `n` currently available GPUs.

        Yields list of `n` ids for available GPUs which are currently not in use.
        If GPUs are available but currently in use, waits until `n` GPUs are free.
        If there are no GPUs available (e.g. no cuda device visible) yields `None` instead.
        The yielded GPU ids are not available for others while the context is up.
        """
        gpus = None
        if self.n_available_gpus > 0:
            assert self.n_available_gpus >= n, 'Requested more gpus per process than available'
            gpus = [self.queue.get() for _ in range(n)]
        yield gpus
        if gpus is not None:
            for idx in gpus:
                self.queue.put(idx)

