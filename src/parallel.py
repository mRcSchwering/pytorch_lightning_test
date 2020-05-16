"""Tools for parallelizing trainings"""
from contextlib import contextmanager
import multiprocessing
from multiprocessing import Process, Manager
from src.config import N_GPUS


def subprocess(target: callable, kwargs: dict) -> dict:
    """
    Run target function in another process.
    Target must accept `**kwargs` and return a dict.

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


class NoDaemonProcess(multiprocessing.Process):
    """A Process that has 'daemon' attribute always return False"""
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(multiprocessing.pool.Pool):
    """
    Extends multiprocessing.pool.Pool with NonDaemonProcess.
    You need this to start a pytorch `DataLoader(..., num_workers=4)` in the pool.

    Note: Sometimes I haved experiences weird behaviour with this.
    E.g. the last execution never finishes. Maybe the parent doesnt recieve
    the signal that all children are done. Anyway, it might help to use
    `subprocess` (see above) in that case.

    Idea from: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    """
    Process = NoDaemonProcess

    def __init__(self, processes: int):
        super().__init__(processes=processes)


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


def subprocess_with_gpus(
        target: callable, queue: GpuQueue,
        kwargs: dict = {}, n_gpus_per_trial: int = 1):
    """
    Convenience function that wraps `target` function in `subprocess` and GPU context.

    `target` must accept arguments in `kwargs` and `gpus` as argument.
    `gpus` will have the list of GPU ids (or `None` if no GPU is available).
    Must return a dictionary with float `loss`.
    """
    kwargs = kwargs.copy()
    with queue.n_gpus(n=n_gpus_per_trial) as gpus:
        kwargs.update({'gpus': gpus})
        res = subprocess(target=target, kwargs=kwargs)
    return res['loss']
