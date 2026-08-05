"""Queue manager: fan out a depth's cases to GPU workers with resumable checkpoints."""

import json
import multiprocessing as mp
import os

from . import config
from .checkpoint import AsyncCheckpointSaver
from .editing import get_mode_logic
from .worker import persistent_gpu_worker


def process_mode_with_checkpoint(depth, mode_key, alg_name, hparams_path, target_gpus, model):
    """Evaluate one (depth, mode) for one (algorithm, model) across `target_gpus`.

    Reads results/chain/<dataset>_seer-kg_depth-{depth}_chains.json and writes
    results/eval/<model>/<alg>/<mode>/results_depth-{depth}.json (with a resumable
    checkpoint).
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    mode_logic = get_mode_logic(mode_key)
    input_file = config.chain_input_path(depth)
    mode_dir = os.path.join(config.EVAL_RESULTS_DIR, model, alg_name, mode_key)
    os.makedirs(mode_dir, exist_ok=True)

    final_output = os.path.join(mode_dir, f"results_depth-{depth}.json")
    checkpoint_file = os.path.join(mode_dir, f"checkpoint_depth-{depth}.json")

    if os.path.exists(final_output):
        print(f"[MANAGER] Results already exist: {final_output}")
        return
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        return

    with open(input_file, "r") as f:
        original_data = json.load(f)

    processed_results = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_results = json.load(f)

    start_idx = len(processed_results)
    total_tasks = len(original_data) - start_idx
    if total_tasks <= 0:
        print("[MANAGER] All tasks already processed in checkpoint.")
        return

    print(f"\n[MANAGER] Resuming {mode_key} (Depth {depth}) at index {start_idx}")

    task_queue, result_queue = mp.Queue(), mp.Queue()
    saver = AsyncCheckpointSaver()
    workers = []

    for gpu_id in target_gpus:
        p = mp.Process(
            target=persistent_gpu_worker,
            args=(gpu_id, alg_name, hparams_path, mode_logic, task_queue, result_queue),
        )
        p.daemon = True
        p.start()
        workers.append(p)

    for i in range(start_idx, len(original_data)):
        task_queue.put((i, original_data[i]))

    try:
        completed_count = 0
        while completed_count < total_tasks:
            case_idx, result, err = result_queue.get()
            if result is not None:
                processed_results.append(result)
                saver.save_async(checkpoint_file, processed_results)
            else:
                print(f"[MANAGER] Skipped task {case_idx} due to error: {err}")
            completed_count += 1
            print(f"[MANAGER] Progress: {completed_count}/{total_tasks} completed.")
    except KeyboardInterrupt:
        print("\n[MANAGER] Manual interruption detected. Shutting down gracefully...")
    finally:
        print("[MANAGER] Sending shutdown signals to workers...")
        for _ in workers:
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass

        for w in workers:  # phase 1: graceful
            try:
                w.join(timeout=60)
            except Exception:
                pass

        stuck = [w for w in workers if w.is_alive()]  # phase 2: SIGTERM
        for w in stuck:
            print(f"[MANAGER] Worker pid={w.pid} unresponsive; terminating.")
            try:
                w.terminate()
            except Exception:
                pass
        for w in stuck:
            try:
                w.join(timeout=10)
            except Exception:
                pass

        stuck = [w for w in workers if w.is_alive()]  # phase 3: SIGKILL
        for w in stuck:
            print(f"[MANAGER] Worker pid={w.pid} survived terminate; sending SIGKILL.")
            try:
                w.kill()
            except Exception:
                pass
        for w in stuck:
            try:
                w.join(timeout=5)
            except Exception:
                pass

        saver.stop()

    if len(processed_results) >= len(original_data) - start_idx:
        os.rename(checkpoint_file, final_output)
        print(f"[MANAGER] Completed: {final_output}")
    else:
        print(f"[MANAGER] Partial completion saved to: {checkpoint_file}")
