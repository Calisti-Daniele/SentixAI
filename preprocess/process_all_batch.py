import os
from process_batch import process_batch_file

def process_all_batches(input_folder="batches", output_folder="done_batches", n_threads=10):
    os.makedirs(output_folder, exist_ok=True)
    batch_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".csv")])

    for f in batch_files:
        input_path = os.path.join(input_folder, f)
        output_path = os.path.join(output_folder, f.replace(".csv", "_done.csv"))

        if os.path.exists(output_path):
            print(f"⏭️  {f} già elaborato, skip.")
            continue

        process_batch_file(input_path, output_path, n_threads)


if __name__ == "__main__":
    process_all_batches()
