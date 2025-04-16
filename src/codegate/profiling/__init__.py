import cProfile


def profiled(prefix):
    def wrapper(func):
        import datetime
        import os
        from functools import wraps

        @wraps(func)
        async def inner(*args, **kwargs):

            ts = datetime.datetime.utcnow()
            fname = f"pstats-{prefix}-{ts.strftime('%Y%m%dT%H%M%S%f')}.prof"
            fname = os.path.join(os.path.abspath("."), fname)

            with cProfile.Profile() as pr:
                res = await func(*args, **kwargs)
                print(f"writing to {fname}")
                pr.dump_stats(fname)
                return res

        if os.getenv(f"CODEGATE_PROFILE_{prefix.upper()}"):
            return inner

        return func

    return wrapper
