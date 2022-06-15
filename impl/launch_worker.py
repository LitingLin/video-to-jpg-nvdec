import os
import sys
import multiprocessing


def launch_worker(func, args=(), kwargs={}, timeout=None):
    p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(timeout)
    if p.exitcode != 0:
        if p.exitcode is None:
            try:
                p.kill()
            except Exception:
                pass
        return False
    return True
