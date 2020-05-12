"""
Using the convenience of multiprocessing.Pool with pytorch.

Originally, I wanted to start several parallel pytorch training runs from one python process.
I can do that with os.fork() but multprocessing.Pool is much more convenient.
However, the processes from that are daemonized and cannot themself have children.
Fortunately, I found this solution:
https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
"""
import multiprocessing


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