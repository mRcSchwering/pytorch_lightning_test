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

N_GPUS = 2


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
    # TODO: not allowing setting processes because if the number is wrong, GpuQueue will hang
    Process = NoDaemonProcess

    def __init__(self):
        n_processes = N_GPUS if N_GPUS > 0 else 1
        super().__init__(processes=n_processes)


class GpuQueue:
    """
    Queue for processes to find available GPUs.

    Initialize in parent process, before starting `Pool` of workers.
    Then add it as argument to each worker.
    Use `one_gpu_per_process` as context to get gpu idxs within process.

    :Example:
        def process(gpus: GpuQueue, config: dict = None):
            with gpus.one_gpu_per_process() as gpu_i:
                print(f'Pid{os.getpid()}: config is {config}, using gpu {gpu_i}')
                time.sleep(2)
            return 'a result'
        
        if __name__ == '__main__':
            inputs = ['set1', 'set2', 'set3', 'set4']
            gpus = GpuQueue()
            
            with NonDaemonPool(processes=2) as pool:
                studies = pool.starmap(process, [(gpus, d) for d in inputs])

            print(f'done, results: {studies}')
    """
    # TODO: should also work with no GPUs/ 1 process
    # TODO: does it work if process = 2 but GPUs = 10?
    # maybe better to fix it (see above)
    # how could it work with study.optimize(Objective(), n_jobs=2, n_trials=10)???

    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        self.all_idxs = set(range(N_GPUS))
        self.queue.put(self.all_idxs)
        self.current_idx = None

    @contextmanager
    def one_gpu_per_process(self):
        """Get GPU index"""
        available_idxs = self.queue.get()
        self.current_idx = available_idxs.pop()
        self.queue.put(available_idxs)
        yield self.current_idx
        
        available_idxs = self.queue.get()
        available_idxs.add(self.current_idx)
        self.queue.put(available_idxs)
