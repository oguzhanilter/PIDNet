import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare the KITTI dataset such that every image has \
                                                  the same size. In addition creates the list file \
                                                  in the format of PIDNet')
    
    parser.add_argument('--root_path'  , type=str, required=True)
    parser.add_argument('--images_relative_path', type=str, required=True)
    parser.add_argument('--labels_relative_path', type=str, required=True)  

    args = parser.parse_args()

    return args


def get_min_sizes(args):
    min_x = 9999999
    min_y = 9999999

    path_to_images = os.path.join(args.root_path, args.images_relative_path)
    for image_file in os.listdir( path_to_images ):
        image = cv2.imread(os.path.join(path_to_images, image_file))
        if min_x > image.shape[0]:
            min_x = image.shape[0]
    
        if min_y > image.shape[1]:
            min_y = image.shape[1]

    path_to_labels =  os.path.join(args.root_path, args.labels_relative_path)
    for label_file in os.listdir( path_to_labels ):
        label = cv2.imread(os.path.join(path_to_labels, label_file))
        if min_x > label.shape[0]:
            min_x = label.shape[0]
    
        if min_y > label.shape[1]:
            min_y = label.shape[1]
        
    return min_x, min_y


def find_matching_label(labels_file_dict: dict, image_file: str):

    labels_file_list = labels_file_dict.keys()
    image_id = image_file.split('.')[0]

    for label_file in labels_file_list:
        if image_id in label_file:
            return label_file

    return None

def prepare_dataset(args):
    min_x, min_y = get_min_sizes(args)

    labels_file_dict = {}
   
    path_to_labels =  os.path.join(args.root_path, args.labels_relative_path)
    for label_file in os.listdir( path_to_labels ):
        labels_file_dict[label_file] = None
        label_abs_path = os.path.join(path_to_labels, label_file)
        label = cv2.imread( label_abs_path )
        cropped_label = label[:min_x, :min_y]
        cv2.imwrite(label_abs_path,cropped_label)
    
    path_to_images = os.path.join(args.root_path, args.images_relative_path)
    for image_file in os.listdir( path_to_images ):

        corresponding_label_file = find_matching_label(labels_file_dict, image_file)

        if corresponding_label_file:
            labels_file_dict[corresponding_label_file] = image_file

            image_abs_path = os.path.join(path_to_images, image_file)
            image = cv2.imread( image_abs_path )
            cropped_image = image[:min_x, :min_y]
            cv2.imwrite(image_abs_path,cropped_image)

        else:
            print(f'The image {image_file} has no corresponding label.')
    
    return labels_file_dict


def create_list_file(args, labels_file_dict : dict):

    train_file = os.path.join( os.getcwd(), 'train.lst' ) 
    with open(train_file, 'w') as f:
        for label, image in labels_file_dict.items():
            if image is not None:
                
                image_relative_path = os.path.join( args.images_relative_path , image)
                label_relative_path = os.path.join( args.labels_relative_path , image)

                f.write(f'{image_relative_path}\t{label_relative_path}\n')
            else:
                print(f'The image {label} has no corresponding image. It will not be added to list.')

if __name__ == '__main__':
    args = parse_args()

    print(f'The path to the directory of images: {args.images_relative_path}')
    print(f'The path to the directory of labels: {args.labels_relative_path}')

    print("Starting croping ... ")
    labels_file_dict = prepare_dataset(args)
    print("Starting creating list file ...")
    create_list_file(args, labels_file_dict)
    print("Done")
