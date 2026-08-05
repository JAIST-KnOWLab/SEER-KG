"""Persistent per-GPU worker process that holds the model in VRAM.

Each worker loads the editor/model once, then consumes cases from a shared task
queue and evaluates every chain in each case, pushing annotated results back.
"""

import os
import time
import traceback
from threading import Thread


def _parent_watchdog(parent_pid, gpu_id):
    """Self-destruct if the manager dies, to release VRAM held by orphaned workers."""
    while True:
        time.sleep(5)
        if os.getppid() != parent_pid:
            print(f"[WORKER-{gpu_id}] Manager (pid={parent_pid}) died; exiting to release VRAM.", flush=True)
            os._exit(1)


def persistent_gpu_worker(gpu_id, alg_name, hparams_path, mode_logic, task_queue, result_queue):
    """Isolated worker: load once, evaluate many cases. Heavy imports are done here
    (inside the spawned process) so the parent stays light."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from . import editing

    Thread(target=_parent_watchdog, args=(os.getppid(), gpu_id), daemon=True).start()

    internal_device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    print(f"\n[WORKER-{gpu_id}] Booting up. Locked to physical GPU-{gpu_id}.")

    t, m, e, train_ds = None, None, None, None
    try:
        t, m, e = editing.load_fresh_components(alg_name, hparams_path, internal_device, str(gpu_id))
        train_ds = editing.get_train_dataset_for_algorithm(alg_name)
        print(f"[WORKER-{gpu_id}] Model loaded successfully. Awaiting tasks...")

        processed_count = 0
        while True:
            task = task_queue.get()
            if task is None:
                break  # poison pill

            case_idx, case = task
            start_time = time.time()
            try:
                effects = case.get("effects", {})
                for etype in ["chains", "onto_chains"]:
                    chains_list = effects.get(etype)
                    if not chains_list:
                        continue
                    for c_info in chains_list:
                        c_info["detail"] = editing.run_mode(
                            t, m, e, c_info["detail"],
                            alg_name=alg_name, train_ds=train_ds, mode=mode_logic,
                        )
                case["processing_time_seconds"] = round(time.time() - start_time, 2)
                result_queue.put((case_idx, case, None))
            except Exception as err:
                print(f"[WORKER-{gpu_id}] Task {case_idx} failed: {err}")
                traceback.print_exc()
                result_queue.put((case_idx, None, str(err)))

            processed_count += 1
            if processed_count % 10 == 0:
                editing.cleanup_gpu(internal_device)

    except Exception as err:
        print(f"[WORKER-{gpu_id}] FATAL SETUP ERROR: {err}")
        traceback.print_exc()
    finally:
        print(f"[WORKER-{gpu_id}] Shutting down and releasing VRAM...")
        del m, e, t, train_ds
        from . import editing as _editing
        _editing.cleanup_gpu(internal_device)
