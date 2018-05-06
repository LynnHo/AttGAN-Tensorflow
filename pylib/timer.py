from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import timeit


class Timer(object):
    """A timer as a context manager.

    Modified from https://github.com/brouberol/contexttimer/blob/master/contexttimer/__init__.py.

    Wraps around a timer. A custom timer can be passed
    to the constructor. The default timer is timeit.default_timer.

    Note that the latter measures wall clock time, not CPU time!
    On Unix systems, it corresponds to time.time.
    On Windows systems, it corresponds to time.clock.

    Keyword arguments:
        is_output -- if True, print output after exiting context.
        format -- 'ms', 's' or 'datetime'
    """

    def __init__(self, timer=timeit.default_timer, is_output=True, fmt='s'):
        assert fmt in ['ms', 's', 'datetime'], "`fmt` should be 'ms', 's' or 'datetime'!"
        self._timer = timer
        self._is_output = is_output
        self._fmt = fmt

    def __enter__(self):
        """Start the timer in the context manager scope."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Set the end time."""
        if self._is_output:
            print(str(self))

    def __str__(self):
        if self._fmt != 'datetime':
            return '%s %s' % (self.elapsed, self._fmt)
        else:
            return str(self.elapsed)

    def start(self):
        self.start_time = self._timer()

    @property
    def elapsed(self):
        """Return the current elapsed time since start."""
        e = self._timer() - self.start_time

        if self._fmt == 'ms':
            return e * 1000
        elif self._fmt == 's':
            return e
        elif self._fmt == 'datetime':
            return datetime.timedelta(seconds=e)


def timer(**timer_kwargs):
    """Function decorator displaying the function execution time.

    All kwargs are the arguments taken by the Timer class constructor.
    """
    # store Timer kwargs in local variable so the namespace isn't polluted
    # by different level args and kwargs

    def wrapped_f(f):
        def wrapped(*args, **kwargs):
            fmt = '[*] function "%(function_name)s" execution time: %(execution_time)s [*]'
            with Timer(**timer_kwargs) as t:
                out = f(*args, **kwargs)
            context = {'function_name': f.__name__, 'execution_time': str(t)}
            print(fmt % context)
            return out
        return wrapped

    return wrapped_f

if __name__ == "__main__":
    import time

    # 1
    print(1)
    with Timer() as t:
        time.sleep(1)
        print(t)
        time.sleep(1)

    with Timer(fmt='datetime') as t:
        time.sleep(1)

    # 2
    print(2)
    t = Timer(fmt='ms')
    t.start()
    time.sleep(2)
    print(t)

    t = Timer(fmt='datetime')
    t.start()
    time.sleep(1)
    print(t)

    # 3
    print(3)

    @timer(fmt='ms')
    def blah():
        time.sleep(2)

    blah()
