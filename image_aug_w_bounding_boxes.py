"""
The script augment images with bounding boxes for object detection applications.
The input annotation files are expected to be in PASCAL xml format.
"""

import numpy as np
import os
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import xml.etree.ElementTree as ET
from PIL import Image
import argparse
from pathlib import Path



def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    data = np.expand_dims(data, axis=0)
    return data


def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "RGB" )
    img.save( outfilename )


def plot_bb(img_aug, bbs_aug):
	bbsoi = ia.BoundingBoxesOnImage(bbs_aug, shape=img_aug.shape)
	image_with_bbs = bbsoi.draw_on_image(img_aug, color=(255, 0, 0), size=2)
	# ia.imshow(image_with_bbs)

	return image_with_bbs


def read_xml_content(file):
	tree = ET.parse(file)
	root = tree.getroot()
	list_with_all_boxes = []

	filename = root.find('filename').text
	for boxes in root.iter('object'):
	    ymin, xmin, ymax, xmax = None, None, None, None

	    for box in boxes.findall("bndbox"):
	        ymin = int(box.find("ymin").text)
	        xmin = int(box.find("xmin").text)
	        ymax = int(box.find("ymax").text)
	        xmax = int(box.find("xmax").text)

	    list_with_single_boxes = [xmin, ymin, xmax, ymax]
	    list_with_all_boxes.append(list_with_single_boxes)

	return filename, list_with_all_boxes

def write_xml_content(file, outputfile, bbs_aug):
    tree = ET.parse(file)
    root = tree.getroot()
    list_with_all_boxes = []

    # TODO: update filename to be the corresponding modified name
    filename = root.find('filename').text
    size = root.find('size')
    size.find('width').text = str(FLAGS.output_image_width)
    size.find('height').text = str(FLAGS.output_image_height)
    for i, boxes in enumerate(root.iter('object')):
        ymin, xmin, ymax, xmax = bbs_aug[i].y1, bbs_aug[i].x1, bbs_aug[i].y2, bbs_aug[i].x2

        for box in boxes.findall("bndbox"):
            box.find("ymin").text = str(int(ymin))
            box.find("xmin").text = str(int(xmin))
            box.find("ymax").text = str(int(ymax))
            box.find("xmax").text = str(int(xmax))

    tree.write(outputfile)

def save_augmented_files(image_aug, bbs_aug, file, xml_file_path, img_output_dir, xml_output_dir, postfix):
	# Store augmented images(.png) and augmented bounding boxes (.xml)

	# global img_output_dir
	# global xml_output_dir

	# declare file path
	filename_aug = file.split('.')[0]+ '_' + postfix
	img_aug_file_path = os.path.join(img_output_dir, filename_aug + '.png')
	xml_aug_file_path = os.path.join(xml_output_dir, filename_aug + '.xml')

	# Write annotation file for the augmented image
	write_xml_content(xml_file_path, xml_aug_file_path, bbs_aug)
	# Save augmented image
	image_aug = np.squeeze(image_aug)  # convert 4d to 3d
	save_image( image_aug, img_aug_file_path)

	# # Save augmented image with bounding boxes (Optional. Used only for debugging purpose)
	# image_with_bbs = plot_bb(image_aug, bbs_aug) # Draw bb on the augmented image
	# save_image( image_with_bbs, os.path.join(img_output_dir, 'with_bbs', filename_aug + '.png'))


def apply_augmentation(img_file_path, xml_file_path, file, img_output_dir, xml_output_dir, num_aug=1):
	# Apply augmentation, save augmented images and bounding boxes

	# Read xml file
    name, bbs = read_xml_content(xml_file_path)
    # Read image file
    image = load_image(img_file_path)
    image = image.astype(np.uint8)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    # Define imgaug bounding boxes
    bounding_boxes = list()
    for bb in bbs:
    	bounding_boxes.append(ia.BoundingBox(x1=int(bb[0]), y1=int(bb[1]), x2=int(bb[2]), y2=int(bb[3])))

    for aug_ind in range(1,num_aug+1):
        print('augmentation ', aug_ind)
        # Define desired augmentation
        if aug_ind == 1:
        	seq = iaa.Sequential([
                iaa.Fliplr(p = 0.5),  # apply horizontal flip
                iaa.Flipud(p = 0.5),  # apply vertical flip
                ])
        elif aug_ind == 2:
            seq = iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale=0.05*255), # apply additive gaussian noise from a normal distribution between (0,0.05*255)
                ])
        elif aug_ind == 3:
        	seq = iaa.Sequential([
                iaa.Rot90(1),     # apply clockwise rotation of 90 degrees keeping the same image size
                ])
        elif aug_ind == 4:
        	seq = iaa. Sequential([
        	    iaa.Rot90(2),     # apply clockwise rotation of 180 degrees keeping the same image size
        	    ])
        elif aug_ind == 5:
            seq = iaa. Sequential([
                iaa.Rot90(3),     # apply clockwise rotation of 270 degrees keeping the same image size
                ])

        # Apply augmentation to image and bounding_boxes
        image_aug, bbs_aug = seq(images=image, bounding_boxes=bounding_boxes)
        seq = iaa. Sequential([
            # sometimes(iaa.Crop(percent=(0, 0.1))),
            iaa.Resize({"height": FLAGS.output_image_height, "width": FLAGS.output_image_width}),    # apply clockwise rotation of 270 degrees keeping the same image size
        ])
        image_aug, bbs_aug = seq(images=image_aug, bounding_boxes=bbs_aug)

        postfix = str(aug_ind)
        save_augmented_files(image_aug, bbs_aug, file, xml_file_path, img_output_dir, xml_output_dir, postfix)

    return image_aug, bbs_aug


def main(FLAGS):

    img_input_dir = FLAGS.input_image_dir
    xml_input_dir = FLAGS.input_bbox_dir

    if not os.path.exists(img_input_dir):
    	raise Exception("Input image directory not correctly provided")

    if not os.path.exists(xml_input_dir):
    	raise Exception("Input annotation directory directory not correctly provided")

    # TODO: Add file check, if no images or wrong format image.
    print('input image dir, input annotation dir', img_input_dir, xml_input_dir)

    img_output_dir = FLAGS.output_image_dir
    Path(img_output_dir).mkdir(parents=True, exist_ok=True)

    xml_output_dir = FLAGS.output_bbox_dir
    Path(xml_output_dir).mkdir(parents=True, exist_ok=True)
    valid_extensions = ['.png', '.jpg', 'jpeg', '.bmp']
    for root, dirs, files in os.walk(img_input_dir):
    	for file in files:
    		# print(file)
    		if file.endswith(tuple(valid_extensions)):
    			print('Applying agumation on image file', file)
    			img_file_path = os.path.join(root, file)
    			xml_file_path = os.path.join(xml_input_dir, file.split('.')[0]+'.xml')

    			num_aug=5
    			image_aug, bbs_aug = apply_augmentation(img_file_path, xml_file_path, file, img_output_dir, xml_output_dir, num_aug)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--input_image_dir',
      type=str,
      default='/home/docker/fs1/Deep_Learning/Datasets/acc_counter/test_jpg/images',
      help='Input image (.png) directory'
    )
    parser.add_argument(
      '--input_bbox_dir',
      type=str,
      default='/home/docker/fs1/Deep_Learning/Datasets/acc_counter/test_jpg/annotations',
      help='Input bounding boxs (.xml) directory.'
    )
    parser.add_argument(
         '--output_image_dir',
         type=str,
         default='/home/docker/fs1/Deep_Learning/Datasets/acc_counter/test_jpg/augmented/images/',
         help='Output augmented image (.png) directory.'
    )
    parser.add_argument(
         '--output_bbox_dir',
         type=str,
         default='/home/docker/fs1/Deep_Learning/Datasets/acc_counter/test_jpg/augmented/annotations/',
         help="Output augmented bounding boxs (.xml) directory."
    )
    parser.add_argument(
         '--output_image_height',
         type=int,
         default=600,
         help="Saved output image height."
    )
    parser.add_argument(
         '--output_image_width',
         type=int,
         default=600,
         help="Saved output image width."
    )


    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
