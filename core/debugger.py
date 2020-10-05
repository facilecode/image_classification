from time import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        end = time() - start
        print(func.__name__, " executed in ", end, " secs. FPS-", 1/end)

        return res
    
    return wrapper