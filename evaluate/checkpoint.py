"""Non-blocking checkpoint saving in a background thread."""

import json
import os
from queue import Queue
from threading import Thread


class AsyncCheckpointSaver:
    """Serialize checkpoint writes off the main thread, atomically."""

    def __init__(self):
        self.queue = Queue()
        self.thread = Thread(target=self._save_worker, daemon=True)
        self.thread.start()

    def _save_worker(self):
        while True:
            data = self.queue.get()
            if data is None:
                break
            filepath, results = data
            try:
                # Write to a temp file first to avoid corruption on interruption.
                temp_path = filepath + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(results, f, indent=2)
                os.replace(temp_path, filepath)
            except Exception as exc:
                print(f"[SAVER] Checkpoint save error: {exc}")

    def save_async(self, filepath, results):
        self.queue.put((filepath, results))

    def stop(self):
        while not self.queue.empty():
            pass
        self.queue.put(None)
        self.thread.join()
