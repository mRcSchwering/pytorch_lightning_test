"""
Using the convenience of multiprocessing.Pool with pytorch.

Originally, I wanted to start several parallel pytorch training runs from one python process.
I can do that with os.fork() but multprocessing.Pool is much more convenient.
However, the processes from that are daemonized and cannot themself have children.
Fortunately, I found this solution:
https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
"""
from contextlib import contextmanager
import multiprocessing
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


class GpuQueue:
    """
    Queue for processes or threads to find available GPUs.

    Initialize in parent process, before starting `Pool` of workers.
    Then add it as argument to each worker.
    Use `one_gpu_per_process` as context to get gpu idxs within process.

    :Example:
        def process(gpu_queue: GpuQueue, config: dict):
            with gpus.one_gpu_per_process() as gpu_i:
                print(f'Pid{os.getpid()}: config is {config}, using gpu {gpu_i}')
                time.sleep(2)
            return 'a result'
        
        if __name__ == '__main__':
            inputs = ['set1', 'set2', 'set3', 'set4']
            gq = GpuQueue()
            
            with NonDaemonPool(processes=2) as pool:
                results = pool.starmap(process, [(gq, d) for d in inputs])

            print(f'done, results: {results}')
    
    Note: Doesn't work 100%. With pytorch_lightning a process sometimes hangs.
    I don't know why. This also seems to happen when the queue is not used.
    The stacktrace includes something with `waiter.acquire()`.
    I gave up searching for the reason.
    """

    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        """
        Get index of currently free GPU.
        
        Context manager that yields a GPU index of a GPU which is currently not being used.
        The index is taken from a queue which other threads or processes use as well.
        Each process takes an index from the queue of available GPU indexes
        and puts it back after execution.
        """
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)
