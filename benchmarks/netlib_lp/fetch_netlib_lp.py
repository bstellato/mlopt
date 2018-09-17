#!/usr/bin/env python
import os
import urllib.request
import shutil
import tarfile

OUTPUT_DIR = "lp_data"
OUTPUT_PATH = os.path.join("lp_data", "mps")   # Directory where to store files
DATASET = "ftp://ftp.numerical.rl.ac.uk/pub/cuter/netlib.tar.gz"
DATASET_NAME = "data.tar.gz"


print("Downloading LP data")

# Create directory where to store the wheels
if os.path.exists(OUTPUT_DIR):
    print("Deleting existing %s directory" % OUTPUT_DIR)
    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

# Download all the files
print("Downloading complete dataset...", end='')
urllib.request.urlretrieve(DATASET, 'data.tar.gz')
print("[OK]")


print("Extracting dataset...", end='')
with tarfile.open(DATASET_NAME) as tar:
    tar.extractall()
print("[OK]")
shutil.move('netlib', OUTPUT_PATH)
os.remove(DATASET_NAME)
os.remove(os.path.join(OUTPUT_PATH, "README"))

# Rename all the files to lowercase mps
for f in os.listdir(OUTPUT_PATH):
    file_name = f[:-4].lower()
    os.rename(os.path.join(OUTPUT_PATH, f),
              os.path.join(OUTPUT_PATH, file_name + ".mps"))

# Remove asterisks and empty lines in files
for file_name in os.listdir(OUTPUT_PATH):
    # Read file and keep only lines
    with open(os.path.join(OUTPUT_PATH, file_name), 'r') as f:
        data = f.read()
        data_stripped = ""
        # Split lines
        for line in data.splitlines():
            if (not line.startswith('*')) and \
                    ("OBJECT BOUND" not in line) and \
                    (line):
                data_stripped += line + "\n"

    # write string to file
    with open(os.path.join(OUTPUT_PATH, file_name), 'w') as f:
        f.write(data_stripped)
