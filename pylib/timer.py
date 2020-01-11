import datetime
import timeit


class Timer:  # deprecated, use tqdm instead
    """A timer as a context manager.

    Wraps around a timer. A custom timer can be passed
    to the constructor. The default timer is timeit.default_timer.

    Note that the latter measures wall clock time, not CPU time!
    On Unix systems, it corresponds to time.time.
    On Windows systems, it corresponds to time.clock.

    Parameters
    ----------
    print_at_exit : boolean
        If True, print when exiting context.
    format : str
        `ms`, `s` or `datetime`.

    References
    ----------
    - https://github.com/brouberol/contexttimer/blob/master/contexttimer/__init__.py.


    """

    def __init__(self, fmt='s', print_at_exit=True, timer=timeit.default_timer):
        assert fmt in ['ms', 's', 'datetime'], "`fmt` should be 'ms', 's' or 'datetime'!"
        self._fmt = fmt
        self._print_at_exit = print_at_exit
        self._timer = timer
        self.start()

    def __enter__(self):
        """Start the timer in the context manager scope."""
        self.restart()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Print the end time."""
        if self._print_at_exit:
            print(str(self))

    def __str__(self):
        return self.fmt(self.elapsed)[1]

    def start(self):
        self.start_time = self._timer()

    restart = start

    @property
    def elapsed(self):
        """Return the current elapsed time since last (re)start."""
        return self._timer() - self.start_time

    def fmt(self, second):
        if self._fmt == 'ms':
            time_fmt = second * 1000
            time_str = '%s %s' % (time_fmt, self._fmt)
        elif self._fmt == 's':
            time_fmt = second
            time_str = '%s %s' % (time_fmt, self._fmt)
        elif self._fmt == 'datetime':
            time_fmt = datetime.timedelta(seconds=second)
            time_str = str(time_fmt)
        return time_fmt, time_str


def timeit(run_times=1, **timer_kwargs):
    """Function decorator displaying the function execution time.

    All kwargs are the arguments taken by the Timer class constructor.

    """
    # store Timer kwargs in local variable so the namespace isn't polluted
    # by different level args and kwargs

    def decorator(f):
        def wrapper(*args, **kwargs):
            timer_kwargs.update(print_at_exit=False)
            with Timer(**timer_kwargs) as t:
                for _ in range(run_times):
                    out = f(*args, **kwargs)
            fmt = '[*] Execution time of function "%(function_name)s" for %(run_times)d runs is %(execution_time)s = %(execution_time_each)s * %(run_times)d [*]'
            context = {'function_name': f.__name__, 'run_times': run_times, 'execution_time': t, 'execution_time_each': t.fmt(t.elapsed / run_times)[1]}
            print(fmt % context)
            return out
        return wrapper

    return decorator


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
    time.sleep(2)
    print(t)

    t = Timer(fmt='datetime')
    time.sleep(1)
    print(t)

    # 3
    print(3)

    @timeit(run_times=5, fmt='s')
    def blah():
        time.sleep(2)

    blah()
