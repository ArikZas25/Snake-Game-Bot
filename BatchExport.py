import os
import glob
from GifExporter import GifExporter


def process_all_replays(output_format: str = 'gif', fps: int = 15) -> None:
    """
    Locates all JSON telemetry files in the replays directory, parses their
    run numbers, and sequentially encodes them into media files.
    """
    # 1. Locate all replay files using pattern matching
    search_pattern = os.path.join("replays", "run_*.json")
    filepaths = glob.glob(search_pattern)

    if not filepaths:
        print("[!] No replays found matching 'replays/run_*.json'.")
        return

    # 2. Extract and sort run numbers mathematically
    run_numbers = []
    for path in filepaths:
        filename = os.path.basename(path)
        # Isolate the integer part of the string
        number_str = filename.replace("run_", "").replace(".json", "")
        try:
            run_numbers.append(int(number_str))
        except ValueError:
            print(f"[Warning] Could not parse run number from {filename}. Skipping.")
            continue

    run_numbers.sort()
    total_runs = len(run_numbers)

    print(f"[*] Found {total_runs} replays. Initiating batch rendering pipeline...")

    # 3. Instantiate the exporter once to minimize Pygame initialization overhead
    exporter = GifExporter()

    # 4. Iterate and export
    for index, run_num in enumerate(run_numbers, start=1):
        print(f"\n[{index}/{total_runs}] Processing Run #{run_num}...")

        # We wrap the call in a try-except block so that if one file is corrupted,
        # the entire batch process does not crash.
        try:
            exporter.export_run(run_number=run_num, fps=fps, output_format=output_format)
        except Exception as e:
            print(f"[Error] Failed to export run {run_num}: {e}")

    print("\n[+] Batch export pipeline completed successfully.")


if __name__ == "__main__":
    # You can change 'gif' to 'mp4' if you prefer smaller file sizes
    process_all_replays(output_format='gif', fps=15)