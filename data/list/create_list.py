import os
"""
# Set the directories to read from
dir1 = '/home/oilter/Documents/SemanticSLAM/PIDNet/data/kitti'

# Get the file paths from the directories

files1 = [os.path.join("train/image", f) for f in os.listdir(dir1 + "/train/image")]
files2 = [os.path.join("train/semantic", f) for f in os.listdir(dir1 + "/train/semantic")]

# Create the output file
output_file = '/home/oilter/Documents/SemanticSLAM/PIDNet/data/list/kitti/train.lst'
with open(output_file, 'w') as f:
  # Iterate through the file paths and write them to the output file
  for file1, file2 in zip(files1, files2):
    f.write(f'{file1}\t{file2}\n')

print(f'File paths written to {output_file}')
"""


import os

# Set the directories to read from
dir1 = '/home/oilter/Documents/SemanticSLAM/PIDNet/data/kitti'

# Get the file paths from the directories

files1 = [os.path.join("test/image", f) for f in os.listdir(dir1 + "/test/image")]
# files2 = [os.path.join("train/semantic", f) for f in os.listdir(dir1 + "/train/semantic")]

# Create the output file
output_file = '/home/oilter/Documents/SemanticSLAM/PIDNet/data/list/kitti/test.lst'
with open(output_file, 'w') as f:
  # Iterate through the file paths and write them to the output file
  for file1 in files1:
    f.write(f'{file1}\n')

print(f'File paths written to {output_file}')