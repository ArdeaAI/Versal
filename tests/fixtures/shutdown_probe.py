"""
Real spawned workers for process-group interrupt regression tests.
"""

import json
import os
import sys
import time
from pathlib import Path

from versal.evolution.evolver import close_assess_pools, create_assess_pool, wait_pool_result
from versal.utils.cancellation import cancelled
from versal.utils.shutdown import EscapeShutdown, ForcedShutdown


def work(args):
    root, busy = args
    Path(root, f"worker-{os.getpid()}").touch()
    if busy == "blocked":
        time.sleep(60)
    if busy:
        while not cancelled():
            time.sleep(0.01)
    return cancelled()


def main():
    root, mode = Path(sys.argv[1]), sys.argv[2]
    with EscapeShutdown():
        try:
            if mode == "startup":
                print("READY", flush=True)
            pool = create_assess_pool(2, str(root / "library"))
            pending = pool.map_async(work, [(str(root), "blocked" if mode == "cleanup" else mode == "busy")] * 2, chunksize=1)
            while len(list(root.glob("worker-*"))) < 2:
                time.sleep(0.01)
                if mode != "busy" and pending.ready():
                    pending = pool.map_async(work, [(str(root), False)] * 2, chunksize=1)
            if mode != "startup":
                print("READY", flush=True)
            while not cancelled():
                time.sleep(0.01)
            print("STOPPING", flush=True)
            if mode == "force":
                time.sleep(60)
            if mode == "cleanup":
                original_join = pool.join

                def joining():
                    print("JOINING", flush=True)
                    return original_join()

                setattr(pool, "join", joining)
                close_assess_pools()
            values = wait_pool_result(pending)
            close_assess_pools()
            print(json.dumps({"stopped": True, "worker_cancelled": values}), flush=True)
        except ForcedShutdown:
            close_assess_pools(force=True)
            raise SystemExit(130) from None


if __name__ == "__main__":
    main()
