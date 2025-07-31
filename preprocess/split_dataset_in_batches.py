import pandas as pd
import os


def split_dataset_in_batches(input_file, batch_size=5000, output_folder="batches"):
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(input_file, sep=";")

    total = len(df)
    print(f"📦 Suddividendo {total} righe in batch da {batch_size}...")

    for i in range(0, total, batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_df.to_csv(f"{output_folder}/batch_{i // batch_size + 1}.csv", index=False, sep=";")

    print(f"✅ {((total - 1) // batch_size) + 1} batch salvati nella cartella '{output_folder}'")


if __name__ == "__main__":
    split_dataset_in_batches("../preprocessed_dataset.csv")
