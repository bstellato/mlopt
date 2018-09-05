#!/usr/bin/env python
#  from bs4 import BeautifulSoup
import os
import urllib.request
import shutil
import tarfile
#  import requests

OUTPUT_DIR = "lp_data"   # Directory where to store the wheels
DATASET = "ftp://ftp.numerical.rl.ac.uk/pub/cuter/netlib.tar.gz"
DATASET_NAME = "data.tar.gz"

#  r = requests.get(WEBSITE)
#  data = r.text
#  soup = BeautifulSoup(data, "html.parser")

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
os.rename('netlib', OUTPUT_DIR)
os.remove(DATASET_NAME)

# Rename all the files to lowercase mps
for f in os.listdir(OUTPUT_DIR):
    file_name = f[:-4].lower()
    os.rename(os.path.join(OUTPUT_DIR, f),
              os.path.join(OUTPUT_DIR, file_name + ".mps"))


# Remove asterisks and empty lines in files
for file_name in os.listdir(OUTPUT_DIR):
    # Read file and keep only lines
    with open(os.path.join(OUTPUT_DIR, file_name), 'r') as f:
        data = f.read()
        data_stripped = ""
        # Split lines
        for line in data.splitlines():
            if (not line.startswith('*')) and \
                    ("OBJECT BOUND" not in line) and \
                    (line):
                data_stripped += line + "\n"

    # write string to file
    with open(os.path.join(OUTPUT_DIR, file_name), 'w') as f:
        f.write(data_stripped)
