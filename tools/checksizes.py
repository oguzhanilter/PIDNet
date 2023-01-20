import os
from PIL import Image

# Set the directory containing the images
image_dir = '/cluster/scratch/oilter/data/kitti/val/image'

# Initialize an empty set to store the sizes
sizes = set()

# Iterate through the images in the directory
for image_file in os.listdir(image_dir):
    # Load the image
    image = Image.open(os.path.join(image_dir, image_file))
    
    # Get the size of the image
    size = image.size
    
    # Add the size to the set
    sizes.add(size)

print(f'Number of unique sizes: {len(sizes)}')
print(f'Sizes: {sizes}')