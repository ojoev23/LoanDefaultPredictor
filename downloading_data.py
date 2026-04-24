import kagglehub
import pandas as pd
import os
import glob

# 1. Download the dataset folder
print("Downloading dataset...")
path = kagglehub.dataset_download("wordsforthewise/lending-club")
print("Base path to dataset files:", path)

# 2. Dynamically search for the CSV (or compressed CSV) file
print("Searching for the 'accepted' CSV file...")

# Look for any file containing 'accepted' and ending in .csv or .csv.gz
search_pattern = os.path.join(path, '**', '*accepted*.csv*')
found_paths = glob.glob(search_pattern, recursive=True)

# CRITICAL FIX: Filter out directories! Only keep actual files.
found_files = [f for f in found_paths if os.path.isfile(f)]

if not found_files:
    print("Error: Could not find the actual CSV file. Let's see what downloaded:")
    for root, dirs, files in os.walk(path):
        for name in files:
            print(os.path.join(root, name))
else:
    # Take the first matched actual file
    full_csv_path = found_files[0]
    print(f"Success! Found the actual data FILE here: {full_csv_path}")

    # 3. Load the data into Pandas
    print("Loading CSV into Pandas (this might take a minute due to file size)...")
    # Pandas read_csv is smart enough to automatically decompress .gz files!
    df = pd.read_csv(full_csv_path, low_memory=False)

    # 4. Verify it worked
    print(f"Dataset loaded! Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

    # 5. Create the manageable Portfolio file
    print("Creating the sampled dataset...")
    df_sampled = df.sample(n=100000, random_state=42)
    df_sampled.to_csv("lending_club_portfolio_data.csv", index=False)
    print("Success! Saved 'lending_club_portfolio_data.csv' for your project!")