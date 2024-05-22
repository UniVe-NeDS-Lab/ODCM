import os
import pickle
import networkx as nx
from scipy.stats import t
from line_profiler import LineProfiler


def giant_component(g: nx.Graph) -> nx.Graph:
    return nx.subgraph(g, max(nx.connected_components(g), key=len))

def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

def flatten(l):
    flat_list = []
    for sublist in l:
        flat_list += sublist
    return flat_list

def ci(series):
    return series.sem()*t.ppf((1 + 0.95) / 2., len(series)-1)

def to_csv_comment(df, path):
    with open(path, 'w') as fr:
        fr.write('#'+df.to_csv())

def cached(cachefile):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """
    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(*args, **kwargs):   # define a wrapper that will finally call "fn" with all arguments            
            # if cache exists -> load it and return its content
            if os.path.exists(cachefile):
                    with open(cachefile, 'rb') as cachehandle:
                        print("using cached result from '%s'" % cachefile)
                        return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            with open(cachefile, 'wb') as cachehandle:
                print("saving result to cache '%s'" % cachefile)
                pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator   # return this "customized" decorator that uses "cachefile"



profiler = LineProfiler()

def profile(func):
    def inner(*args, **kwargs):
        profiler.add_function(func)
        profiler.enable_by_count()
        return func(*args, **kwargs)
    return inner
