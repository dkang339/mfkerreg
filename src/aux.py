from joblib.externals.loky import get_reusable_executor
import signal
import atexit
import sys

# clean up loky
def cleanup_parallel():
    try:
        get_reusable_executor().shutdown(wait=False, kill_workers=True)
    except Exception:
        pass

def signal_handler(signum, frame):
    cleanup_parallel()
    sys.exit(1)

def cleanup_loky():
    get_reusable_executor().shutdown(wait=False)
    # register cleanup handlers
    atexit.register(cleanup_parallel)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)