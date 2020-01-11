import concurrent.futures
import functools
import multiprocessing


def run_parallels(work_fn, iterable, max_workers=None, chunksize=1, processing_bar=True, backend_executor=multiprocessing.Pool, debug=False):
    if not debug:
        with backend_executor(max_workers) as executor:
            try:
                works = executor.imap(work_fn, iterable, chunksize=chunksize)  # for multiprocessing.Pool
            except:
                works = executor.map(work_fn, iterable, chunksize=chunksize)

            if processing_bar:
                try:
                    import tqdm
                    try:
                        total = len(iterable)
                    except:
                        total = None
                    works = tqdm.tqdm(works, total=total)
                except ImportError:
                    print('`import tqdm` fails! Run without processing bar!')

            results = list(works)
    else:
        results = [work_fn(i) for i in iterable]
    return results

run_parallels_mp = run_parallels
run_parallels_cfprocess = functools.partial(run_parallels, backend_executor=concurrent.futures.ProcessPoolExecutor)
run_parallels_cfthread = functools.partial(run_parallels, backend_executor=concurrent.futures.ThreadPoolExecutor)


if __name__ == '__main__':
    import time

    def work(i):
        time.sleep(0.0001)
        i**i
        return i

    t = time.time()
    results = run_parallels_mp(work, range(10000), max_workers=2, chunksize=1, processing_bar=True, debug=False)
    for i in results:
        print(i)
    print(time.time() - t)
