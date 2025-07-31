import pandas as pd
import os

folder = "done_batches"
dfs = [pd.read_csv(os.path.join(folder, f), sep=";") for f in sorted(os.listdir(folder)) if f.endswith("_done.csv")]
final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv("ready_dataset_for_training.csv", sep=";", index=False)
