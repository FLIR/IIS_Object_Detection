import json
import os
import glob
import xml.etree.ElementTree as ET


def ope(path_to_check):
    return os.path.exists(path_to_check)


class Config(object):
    ABSOLUTE_DATASETS_PATH = ""
    DATASET_FODLER = ""
    DATASET_IDENTIFIER = ""
    IMAGE_FOLDER_NAME = ""
    ANNOTAION_FOLDERNAME = ""
    IMAGE_EXTENSION = ""

    PROJECT_NAME = ""
    TEST_CLASSES = []
    TEST_SET_PERCENTAGE = 1
    ABSOLUTE_PATH_NETWORK = ""
    PRETRAINED_NETWORK_FILE = ""
    ABSOLUTE_OUTPUT_PROJECT_PATH = ""
    PATH_TO_CONVERT_ANNOSET_DOT_EXE = ""
    PATH_TO_GET_IMAGE_SIZE_DOT_EXE = ""
    PATH_TO_CAFFE_DOT_EXE = ""
    CONTINUE_TRAINING = False
    CAFFE_EXECUTION_COUNT = 0

    def __init__(self,
                 abs_datasets_path: str,
                 dataset_folder: str,
                 dataset_id: str,
                 img_folder_name: str,
                 anno_folder_name: str,
                 img_ext: str,
                 classes: list,
                 proj: str,
                 test_percentage: int,
                 abs_network_path: str,
                 pretrained_network: str,
                 abs_output_path: str,
                 annoset_exe: str,
                 image_size_exe: str,
                 caffe_exe: str,
                 cont: bool = False):
        super(Config, self).__init__()
        self.ABSOLUTE_DATASETS_PATH = abs_datasets_path
        self.DATASET_FODLER = dataset_folder
        self.DATASET_IDENTIFIER = dataset_id
        self.IMAGE_FOLDER_NAME = img_folder_name
        self.ANNOTAION_FOLDERNAME = anno_folder_name
        self.IMAGE_EXTENSION = img_ext

        self.TEST_CLASSES = classes
        self.PROJECT_NAME = proj
        self.TEST_SET_PERCENTAGE = test_percentage

        self.ABSOLUTE_PATH_NETWORK = abs_network_path
        self.PRETRAINED_NETWORK_FILE = pretrained_network

        self.ABSOLUTE_OUTPUT_PROJECT_PATH = abs_output_path
        self.PATH_TO_CONVERT_ANNOSET_DOT_EXE = annoset_exe
        self.PATH_TO_GET_IMAGE_SIZE_DOT_EXE = image_size_exe
        self.PATH_TO_CAFFE_DOT_EXE = caffe_exe
        self.CONTINUE_TRAINING = cont
        self.CAFFE_EXECUTION_COUNT = 0

    def print_config(self):
        print("ABSOLUTE_DATASETS_PATH ", self.ABSOLUTE_DATASETS_PATH)
        print("DATASET_FODLER ", self.DATASET_FODLER)
        print("DATASET_IDENTIFIER ", self.DATASET_IDENTIFIER)
        print("IMAGE_FOLDER_NAME ", self.IMAGE_FOLDER_NAME)
        print("ANNOTAION_FOLDERNAME ", self.ANNOTAION_FOLDERNAME)
        print("IMAGE_EXTENSION ", self.IMAGE_EXTENSION)

        print("PROJECT_NAME ", self.PROJECT_NAME)
        print("TEST_CLASSES ", self.TEST_CLASSES)
        print("TEST_SET_PERCENTAGE ", self.TEST_SET_PERCENTAGE)
        print("ABSOLUTE_PATH_NETWORK ", self.ABSOLUTE_PATH_NETWORK)
        print("PRETRAINED_NETWORK_FILE ", self.PRETRAINED_NETWORK_FILE)
        print("ABSOLUTE_OUTPUT_PROJECT_PATH ", self.ABSOLUTE_OUTPUT_PROJECT_PATH)
        print("PATH_TO_CONVERT_ANNOSET_DOT_EXE ", self.PATH_TO_CONVERT_ANNOSET_DOT_EXE)
        print("PATH_TO_GET_IMAGE_SIZE_DOT_EXE ", self.PATH_TO_GET_IMAGE_SIZE_DOT_EXE)
        print("PATH_TO_CAFFE_DOT_EXE ", self.PATH_TO_CAFFE_DOT_EXE)

    @staticmethod
    def read_json_to_config(json_file):
        with open(json_file) as json_file:
            the_dict = json.load(json_file)
            configuration = Config(
                abs_datasets_path=the_dict["ABSOLUTE_DATASETS_PATH"],
                dataset_folder=the_dict.get("DATASET_FOLDER", ""),
                dataset_id=the_dict.get("DATASET_IDENTIFIER", ""),
                img_folder_name=the_dict.get("IMAGE_FOLDER_NAME", ""),
                anno_folder_name=the_dict.get("ANNOTAION_FOLDERNAME", ""),
                img_ext=the_dict["IMAGE_EXTENSION"],
                proj=the_dict["PROJECT_NAME"],
                classes=the_dict["TEST_CLASSES"],
                test_percentage=the_dict["TEST_SET_PERCENTAGE"],
                abs_network_path=the_dict["ABSOLUTE_PATH_NETWORK"],
                pretrained_network=the_dict["PRETRAINED_NETWORK_FILE"],
                abs_output_path=the_dict["ABSOLUTE_OUTPUT_PROJECT_PATH"],
                annoset_exe=the_dict["PATH_TO_CONVERT_ANNOSET_DOT_EXE"],
                image_size_exe=the_dict["PATH_TO_GET_IMAGE_SIZE_DOT_EXE"],
                caffe_exe=the_dict["PATH_TO_CAFFE_DOT_EXE"])
            configuration.CAFFE_EXECUTION_COUNT = the_dict["CAFFE_EXECUTION_COUNT"]
            configuration.CONTINUE_TRAINING = the_dict["CONTINUE_TRAINING"]
            return configuration

    def dump_json_to_file(self, target_file):
        with open(target_file, 'w') as fp:
            dmp_string = json.dumps(self.__dict__, indent=4)
            fp.write(dmp_string)

    def check_paths_exists(self):
        data_folder = \
            os.path.join(self.ABSOLUTE_DATASETS_PATH,
                         self.DATASET_FODLER,
                         self.DATASET_IDENTIFIER)
        image_folder = os.path.join(data_folder, self.IMAGE_FOLDER_NAME)
        annotation_folder = os.path.join(data_folder, self.ANNOTAION_FOLDERNAME)
        pretrained = os.path.join(self.ABSOLUTE_PATH_NETWORK, self.PRETRAINED_NETWORK_FILE)
        return \
            ope(image_folder) and \
            ope(annotation_folder) and \
            ope(self.ABSOLUTE_PATH_NETWORK) and \
            ope(pretrained) and \
            ope(self.PATH_TO_CONVERT_ANNOSET_DOT_EXE) and \
            ope(self.PATH_TO_GET_IMAGE_SIZE_DOT_EXE) and \
            ope(self.PATH_TO_CAFFE_DOT_EXE)

    def get_classes_from_annotations_from_testtrainval_txt(self, file_paths: list):
        items = []
        for each in file_paths:
            xml_globs = glob.glob("{}/*xml".format(each))
            for each_xml_file in xml_globs:
                tree = ET.parse(each_xml_file)
                root = tree.getroot()
                for child in root:
                    if child.tag == "object":
                        for c2 in child:
                            if c2.tag == "name":
                                new_thing = c2.text
                                if new_thing not in items:
                                    items.append(new_thing)

        self.TEST_CLASSES = list(items)

    def get_classes_from_annotations(self):
        xml_folder = os.path.join(self.ABSOLUTE_DATASETS_PATH,
                                  self.DATASET_FODLER,
                                  self.DATASET_IDENTIFIER,
                                  self.ANNOTAION_FOLDERNAME)
        xml_globs = glob.glob("{}/*xml".format(xml_folder))
        items = []
        for each_xml_file in xml_globs:
            tree = ET.parse(each_xml_file)
            root = tree.getroot()
            for child in root:
                if child.tag == "object":
                    for c2 in child:
                        if c2.tag == "name":
                            new_thing = c2.text
                            if new_thing not in items:
                                items.append(new_thing)

        self.TEST_CLASSES = list(items)

    def get_dataset_path(self):
        return os.path.join(self.ABSOLUTE_DATASETS_PATH, self.DATASET_FODLER, self.DATASET_IDENTIFIER)

    def override_dataset_path(self, the_path):
        self.ABSOLUTE_DATASETS_PATH = the_path
        self.DATASET_FODLER = ""
        self.DATASET_IDENTIFIER = ""


def generate_dummy_config():
    config = Config(
        abs_datasets_path="/base/path/to/datasets/{remove_from_here_to_end}/{dataset_folder}/{dataset_id}",
        dataset_folder="{abs_datasets_path}/{remove_from_start_to_here}only_fill_this_part",
        dataset_id="{dataset_folder}/{remove_from_start_to_here}only_fill_this_part",
        img_folder_name="Images",
        anno_folder_name="Annotations",
        img_ext="jpg",
        proj="dummy",
        classes=["d", "u", "m", "y"],
        test_percentage=25,
        abs_network_path="/path/to/network/",
        pretrained_network="pretrained_weights_name",
        abs_output_path="/path/to/project/output",
        annoset_exe="/path/to/create_annoset.exe",
        image_size_exe="/path/to/get_image_size.exe",
        caffe_exe="/path/to/caffe.exe")
    config.dump_json_to_file("dummy.json")


def generate_default_config():
    config = Config(
        abs_datasets_path="/base/path/to/datasets/{remove_from_here_to_end}/{dataset_folder}/{dataset_id}",
        dataset_folder="{abs_datasets_path}/{remove_from_start_to_here}only_fill_this_part",
        dataset_id="{dataset_folder}/{remove_from_start_to_here}only_fill_this_part",
        img_folder_name="Images",
        anno_folder_name="Annotations",
        img_ext="jpg",
        proj="dummy",
        classes=["d", "u", "m", "y"],
        test_percentage=25,
        abs_network_path="/path/to/network/",
        pretrained_network="pretrained_weights_name",
        abs_output_path="/path/to/project/output",
        annoset_exe="/path/to/create_annoset.exe",
        image_size_exe="/path/to/get_image_size.exe",
        caffe_exe="/path/to/caffe.exe")
    config.dump_json_to_file("dummy.json")


if __name__ == '__main__':
    generate_dummy_config()
