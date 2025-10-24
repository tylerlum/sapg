from pathlib import Path

from recorded_data_scripts.recorded_data import RecordedData

filepath = Path(
    "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-32-39.npz"
)
assert filepath.exists(), f"File {filepath} does not exist"
recorded_data = RecordedData.from_file(filepath)

start = None
end = 310
recorded_data = recorded_data.slice(start=start, end=end)
output_filepath = filepath.parent / f"{filepath.stem}_{start}_{end}.npz"
print(f"Saving sliced recorded data to {output_filepath}")
recorded_data.to_file(output_filepath)
