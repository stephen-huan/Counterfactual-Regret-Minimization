# maintains methods for caching purposes
import pickle, functools, os
try:
    from scipy.stats import entropy
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Scientific libraries not found, some functionality may be broken...")

def get_name(f) -> str:
    """ Makes a filename for a given function. """
    return f"{NAME}_{f.__qualname__}.pickle"

def load(f):
    """ Gets the cached file given a function. """
    with open(get_name(f), "rb") as f:
        return pickle.load(f)

def cache(overwrite: bool=False):
    """ Stores the output of a function in a pickle file. """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not os.path.exists(get_name(f)) or overwrite:
                v = f(*args, **kwargs)
                with open(get_name(f), "wb") as fi:
                    pickle.dump(v, fi)
                return v
            return load(f)

        return wrapper

    return decorator

def graph(f):
    """ Graphs the output of the function. """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        nash = load(f)
        # can either be single player or multiple players
        nash = nash[0] if isinstance(nash, tuple) else nash

        plt.plot(list(map(lambda x: entropy(nash, x), f(*args, graph=True, **kwargs))))
        plt.title("Loss over time")
        plt.ylabel("KL divergence between Nash equilibrium and strategy")
        plt.xlabel("Time (iterations)")
        plt.show()
        exit()

    return wrapper
