import cv2
import xml.etree.ElementTree as etree
from pascal_voc_writer import Writer
import argparse
import shutil
import numpy as np
import os
import glob
import sys
from Config import *

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat

dirsep = ""

if sys.platform == 'linux' or sys.platform == 'darwin':
    dirsep = '/'
elif sys.platform == 'win32':
    dirsep = '\\'


def parse_label_dot_txt(labels_file):
    labels = []
    with open(labels_file, 'r') as lfile:
        lines = lfile.readlines()
        for each_line in lines:
            labels.append(each_line.strip())
    return labels


def parse_test_txt_file(test_file):
    dataset = []
    # change this to dynamically load from config
    config = Config.get_latest_config()
    path_to_dataset = os.path.join(config.ABSOLUTE_DATASETS_PATH, config.DATASET_FODLER, config.DATASET_IDENTIFIER)
    with open(test_file, 'r') as lfile:
        lines = lfile.readlines()
        for each_line in lines:
            image, anno = each_line.strip().split(' ')
            abs_image_path = f"{path_to_dataset}\\{image}"
            abs_anno_path = f"{path_to_dataset}\\{anno}"
            tup = abs_image_path, abs_anno_path
            dataset.append(tup)
    return dataset


def make_dir_if_dne(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def deal_with_single_xml(annotation_xml: str, gt_folder: str):
    tree = etree.parse(open(annotation_xml))
    root = tree.getroot()
    gt_filename = os.path.basename(annotation_xml).split('.')[0]
    gt_filepath = os.path.join(gt_folder, f'{gt_filename}.txt')
    with open(gt_filepath, 'w') as gt_file:
        for i, obj in enumerate(root.iter('object')):
            cls = obj.find('name').text
            bounding_box = obj.find('bndbox')
            line_to_write = (cls + ','
                             + str(int(float(bounding_box.find('xmin').text))) + ','
                             + str(int(float(bounding_box.find('ymin').text))) + ','
                             + str(int(float(bounding_box.find('xmax').text))) + ','
                             + str(int(float(bounding_box.find('ymax').text))) + '\n')
            gt_file.write(line_to_write)


def deal_with_xmls(annotation_dir: str, output_dir: str):
    annotations = glob.glob(f"{annotation_dir}{dirsep}*xml")
    gt_folder = os.path.join(output_dir, 'groundtruths')
    make_dir_if_dne(gt_folder)

    for each_anno in annotations:
        deal_with_single_xml(each_anno, gt_folder)


class BoundingBoxI:
    def __init__(self, lbl: int, conf: float, sx: int, sy: int, nx: int, ny: int):
        super(BoundingBoxI, self).__init__()
        self.label_index = lbl
        self.confidence = conf
        self.start_x = sx
        self.start_y = sy
        self.end_x = nx
        self.end_y = ny


class DetectionEngine:
    # purpose of this is to load the network and initialize the detection object and take images and run
    def __init__(self, lab: list, prototxt: str, caffemodel: str):
        self.labels = parse_label_dot_txt(lab)
        self.colors = np.random.uniform(0, 255, size=(len(self.labels), 3))
        print('[Status] Loading Model...')
        self.nn = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        print('[Status] Model Loaded...')

    def do_single_detect_file(self, file_name, min_confidence):
        frame = cv2.imread(file_name)
        print(file_name)
        return self.do_single_detect(frame, min_confidence)

    def do_single_detect(self, frame, min_confidence):
        # Converting Frame to Blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        # Passing Blob through network to detect and predict
        self.nn.setInput(blob)
        detections = self.nn.forward()
        detections_list = []

        (h, w) = frame.shape[:2]

        for i in np.arange(0, detections.shape[2]):
            # Extracting the confidence of predictions
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                label_index = detections[0, 0, i, 1]
                s_x = detections[0, 0, i, 3] * w
                s_y = detections[0, 0, i, 4] * h
                n_x = detections[0, 0, i, 5] * w
                n_y = detections[0, 0, i, 6] * h
                detections_list.append(BoundingBoxI(label_index, confidence, int(s_x), int(s_y), int(n_x), int(n_y)))
        return detections_list

    def add_bb_to_(self, frame, each_obj: BoundingBoxI):
        label = "{}: {:.2f}%".format(self.labels[int(each_obj.label_index)], each_obj.confidence * 100)
        colour = self.colors[int(each_obj.label_index) % len(self.colors)]
        cv2.rectangle(frame,
                      (each_obj.start_x, each_obj.start_y),
                      (each_obj.end_x, each_obj.end_y), colour, 2)
        cv2.putText(frame, label, (each_obj.start_x, each_obj.start_y),
                    cv2.FONT_HERSHEY_COMPLEX, 1, colour, 1)

    def do_stuff(self, image_dir: str, annotation_dir: str, output_dir: str,
                 metric_output: bool = False, xml_output: bool = False,
                 min_confidence: float = 0.0, image_ext: str = "jpg"):

        images = glob.glob(f"{image_dir}{dirsep}*{image_ext}")
        pred_output_folder = ""
        xml_output_folder = ""
        img_output_folder = ""
        if xml_output:
            xml_output_folder = os.path.join(output_dir, 'xml')
            img_output_folder = os.path.join(output_dir, 'images')
            make_dir_if_dne(xml_output_folder)
            make_dir_if_dne(img_output_folder)
        if metric_output:
            deal_with_xmls(annotation_dir, output_dir)
            pred_output_folder = os.path.join(output_dir, 'predictions')
            make_dir_if_dne(pred_output_folder)

        for each_image in images:
            frame = cv2.imread(each_image)

            # if mono image?
            if frame.shape[2] != 3:
                print(f"mono image? {each_image}")
                height = frame.shape[0]
                width = frame.shape[1]
                frame = np.zeros((height, width, 3), np.uint8)
                # copy to all channels
                image = cv2.imread(each_image, 0)
                frame[:, :, 0] = image
                frame[:, :, 1] = image
                frame[:, :, 2] = image

            results = self.do_single_detect(frame, min_confidence)
            # running through for loop twice redundant might want to improve this
            # create the strings to be written first then dump it if they want it at the end...
            # ram is faster than disk
            pred_dump_string = ""
            path_to_image = each_image
            width = frame.shape[1]
            height = frame.shape[0]
            writer = Writer(path_to_image, width, height)
            for each_obj in results:
                a_line = f"{self.labels[int(each_obj.label_index)]},{each_obj.confidence},{each_obj.start_x}," \
                         f"{each_obj.start_y},{each_obj.end_x},{each_obj.end_y}\n"
                pred_dump_string += a_line
                writer.addObject(self.labels[int(each_obj.label_index)],
                                 each_obj.start_x, each_obj.start_y,
                                 each_obj.end_x, each_obj.end_y)

                # Drawing the prediction and bounding box
                self.add_bb_to_(frame, each_obj)

            if metric_output:
                pred_filename = os.path.basename(each_image).split('.')[0]
                pred_filepath = os.path.join(pred_output_folder, f'{pred_filename}.txt')
                print(f"saving predictions to {pred_filepath}")
                with open(pred_filepath, 'w') as pred_file:
                    pred_file.write(pred_dump_string)

            if xml_output:
                xml_filename = os.path.basename(each_image).split('.')[0]
                xml_filepath = os.path.join(xml_output_folder, f'{xml_filename}.xml')

                writer.save(xml_filepath)
                filename = os.path.basename(each_image).split('.')[0]
                img_fname = f"{filename}.{image_ext}"
                img_filename = os.path.join(img_output_folder, img_fname)
                print(f"saving image to {img_filename}")
                cv2.imwrite(img_filename, frame)

    def use_list(self, output_dir: str, metric_output: bool = False, xml_output: bool = False,
                 min_confidence: float = 0.0, image_ext: str = "jpg"):

        # images = glob.glob(f"{image_dir}{dirsep}*{image_ext}")
        files = parse_test_txt_file("test.txt")
        pred_output_folder = ""
        xml_output_folder = ""
        img_output_folder = ""
        if xml_output:
            xml_output_folder = os.path.join(output_dir, 'xml')
            img_output_folder = os.path.join(output_dir, 'images')
            make_dir_if_dne(xml_output_folder)
            make_dir_if_dne(img_output_folder)

        gt_folder = os.path.join(output_dir, 'groundtruths')
        make_dir_if_dne(gt_folder)
        for each_image_file, annotation_xml in files:
            deal_with_single_xml(annotation_xml, gt_folder)
            pred_output_folder = os.path.join(output_dir, 'predictions')
            make_dir_if_dne(pred_output_folder)
            frame = cv2.imread(each_image_file)

            # if mono image?
            if frame.shape[2] != 3:
                print(f"mono image? {each_image_file}")
                height = frame.shape[0]
                width = frame.shape[1]
                frame = np.zeros((height, width, 3), np.uint8)
                # copy to all channels
                image = cv2.imread(each_image_file, 0)
                frame[:, :, 0] = image
                frame[:, :, 1] = image
                frame[:, :, 2] = image

            results = self.do_single_detect(frame, min_confidence)
            # running through for loop twice redundant might want to improve this
            # create the strings to be written first then dump it if they want it at the end...
            # ram is faster than disk
            pred_dump_string = ""
            width = frame.shape[1]
            height = frame.shape[0]
            writer = Writer(each_image_file, width, height)
            for each_obj in results:
                a_line = f"{self.labels[int(each_obj.label_index)]},{each_obj.confidence},{each_obj.start_x}," \
                         f"{each_obj.start_y},{each_obj.end_x},{each_obj.end_y}\n"
                pred_dump_string += a_line
                writer.addObject(self.labels[int(each_obj.label_index)],
                                 each_obj.start_x, each_obj.start_y,
                                 each_obj.end_x, each_obj.end_y)

                # Drawing the prediction and bounding box
                self.add_bb_to_(frame, each_obj)

            if metric_output:
                pred_filename = os.path.basename(each_image_file).split('.')[0]
                pred_filepath = os.path.join(pred_output_folder, f'{pred_filename}.txt')
                print(f"saving predictions to {pred_filepath}")
                with open(pred_filepath, 'w') as pred_file:
                    pred_file.write(pred_dump_string)

            if xml_output:
                xml_filename = os.path.basename(each_image_file).split('.')[0]
                xml_filepath = os.path.join(xml_output_folder, f'{xml_filename}.xml')

                writer.save(xml_filepath)
                filename = os.path.basename(each_image_file).split('.')[0]
                img_fname = f"{filename}.{image_ext}"
                img_filename = os.path.join(img_output_folder, img_fname)
                print(f"saving image to {img_filename}")
                cv2.imwrite(img_filename, frame)


def get_latest_caffemodel():
    '''this is running from the project folder we can assume that we are ignoring absolute path'''
    snapshot_glob = glob.glob("snapshot/*caffemodel")
    if len(snapshot_glob) < 1:
        return
    else:
        snapshot_glob.sort(key=os.path.getmtime)
        return snapshot_glob[-1]


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(",")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(nameOfImage,
                                 idClass,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coordType,
                                 imgSize,
                                 BBType.GroundTruth,
                                 format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(nameOfImage,
                                 idClass,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coordType,
                                 imgSize,
                                 BBType.Detected,
                                 confidence,
                                 format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


def extra_large_method_to_deal_with_stuff(save_path: str, absolute_path_groundtruth: str, absolute_path_detection: str,
                                          gtFormat=BBFormat.XYWH, detFormat=BBFormat.XYWH,
                                          gtCoordType=CoordinatesType.Absolute, detCoordType=CoordinatesType.Absolute,
                                          imgSize=(0, 0), iouThreshold=0.5, showPlot=False):
    # Clear folder and save results
    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path)

    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(absolute_path_groundtruth,
                                                    True,
                                                    gtFormat,
                                                    gtCoordType,
                                                    imgSize=imgSize)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(absolute_path_detection,
                                                    False,
                                                    detFormat,
                                                    detCoordType,
                                                    allBoundingBoxes,
                                                    allClasses,
                                                    imgSize=imgSize)
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=save_path,
        showGraphic=showPlot)

    f = open(os.path.join(save_path, 'results.txt'), 'w')
    f.write('Object Detection Metrics\n')
    f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    f.write('Average Precision (AP), Precision and Recall per class:')

    # each detection is a class
    for metricsPerClass in detections:

        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            f.write('\n\nClass: %s' % cl)
            f.write('\nAP: %s' % ap_str)
            f.write('\nPrecision: %s' % prec)
            f.write('\nRecall: %s' % rec)

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)


voc = "D:\\Desktop\\caffe_exe\\0_VOC2007\\"
default_image_dir = f"{voc}JPEGImages"
default_anno_dir = f"{voc}Annotations"
default_model = get_latest_caffemodel()
default_output_dir = f"test_{os.path.basename(default_model)}{dirsep}"

# Constructing Argument Parse to input from Command Line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_dir", default=default_image_dir, required=False, help='Path to images')
ap.add_argument("-e", "--image_ext", default="jpg", required=False, help='Image extension')
ap.add_argument("-a", "--annotation_dir", default=default_anno_dir, required=False, help='Path to annotations')
ap.add_argument("-o", "--output_dir", default=default_output_dir, required=False, help='Path to output')
ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt", required=False, help='Path to prototxt')
ap.add_argument("-m", "--model", default=default_model, required=False, help='Path to model weights')
ap.add_argument("-l", "--labels", default="labels.txt", required=False, help='Path to model labels')
ap.add_argument("-c", "--confidence", type=float, default=0.0)
ap.add_argument('--xml_output', type=bool, default=True, help='Save prediction xml and draw bounding boxes on images .')
ap.add_argument('--metric_output', type=bool, default=True,
                help='Save prediction GroundTruth and Predictions csv files.')

FLAGS = ap.parse_args()
de = DetectionEngine(FLAGS.labels,
                     FLAGS.prototxt,
                     FLAGS.model)

# de.do_stuff(FLAGS.image_dir,
#             FLAGS.annotation_dir,
#             FLAGS.output_dir,
#             FLAGS.metric_output,
#             FLAGS.xml_output,
#             FLAGS.confidence,
#             "jpg")

de.use_list(FLAGS.output_dir,
            FLAGS.metric_output,
            FLAGS.xml_output,
            FLAGS.confidence,
            "jpg")

output_path = os.path.abspath(f"{FLAGS.output_dir}{dirsep}results")
absolute_path_groundtruth = os.path.abspath(f"{FLAGS.output_dir}{dirsep}groundtruths")
absolute_path_detection = os.path.abspath(f"{FLAGS.output_dir}{dirsep}predictions")
extra_large_method_to_deal_with_stuff(output_path, absolute_path_groundtruth, absolute_path_detection)
