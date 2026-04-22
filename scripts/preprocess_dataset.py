import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess LeRobot parquet datasets: add cumulative action_states column."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Root directory of the dataset (searched recursively for *.parquet).",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=100,
        help="Parquet row group size (default: 100, suitable for libero).",
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    row_group_size = args.row_group_size

    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset_dir is not a directory: {dataset_dir}")

    for file_path in dataset_dir.rglob("*.parquet"):
        print(f"Processing {file_path} ...")

        table = pq.read_table(file_path)
        actions = np.array(
            table["actions"].to_pylist(),
            dtype=np.float32,
        )
        T = actions.shape[0]

        initial_state = np.array(
            [0, 0, 0, 0, 0, 0, 0], dtype=np.float32
        )  # TODO: replace with the initial state of your own dataset. Zeros is the default initial state for libero dataset.
        action_states = np.concatenate([initial_state[None, :], actions], axis=0)
        action_states = np.cumsum(action_states, axis=0)

        action_states_col = pa.array(action_states.tolist())

        if "action_states" in table.column_names:
            table = table.drop(["action_states"])

        new_table = table.append_column("action_states", action_states_col[:-1])

        pq.write_table(new_table, file_path, row_group_size=row_group_size)
        print(f"Saved {file_path}")

    print("All files processed.")


if __name__ == "__main__":
    main()
