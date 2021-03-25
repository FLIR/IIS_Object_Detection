import io
import random
import subprocess
import shutil
import sys
from Config import *


class Pft(object):

    def __init__(self, config: Config):
        super(Pft, self).__init__()
        self.config = config

    @staticmethod
    def other_class(item_name, item_label_index, item_display_name):
        return_string = 'item { \n' + \
                        'name: \"{}\"\n  '.format(item_name) + \
                        'label: {}\n  '.format(item_label_index) + \
                        'display_name: \"{}\"\n'.format(item_display_name) + \
                        '}\n'
        return return_string

    def generate_labelmap_prototxt_file(self):
        background_class = 'item {\n  name: \"none_of_the_above\"\n  label: 0\n  display_name: \"background\"\n}\n'
        file_dir = os.path.join(self.config.ABSOLUTE_OUTPUT_PROJECT_PATH, self.config.PROJECT_NAME)
        label_map_prototxt_file = "{}/labelmap.prototxt".format(file_dir)
        try:
            os.makedirs(file_dir)
        except FileExistsError:
            print(file_dir, " exists")

        dump_string = ""
        dump_string += background_class
        class_index = 0
        for each_class in self.config.TEST_CLASSES:
            class_index += 1
            dump_string += self.other_class(each_class, class_index, each_class)

        with io.open(label_map_prototxt_file, 'w', newline='\n') as fpwrite:
            fpwrite.write(dump_string)

    def verify_image_and_annotation_folder(self):
        abs_images_path = os.path.join(self.config.ABSOLUTE_DATASETS_PATH,
                                       self.config.DATASET_FODLER,
                                       self.config.DATASET_IDENTIFIER,
                                       self.config.IMAGE_FOLDER_NAME)
        abs_annotations_path = os.path.join(self.config.ABSOLUTE_DATASETS_PATH,
                                            self.config.DATASET_FODLER,
                                            self.config.DATASET_IDENTIFIER,
                                            self.config.ANNOTAION_FOLDERNAME)

        number_of_images = len(glob.glob("{}/*{}".format(abs_images_path, self.config.IMAGE_EXTENSION)))
        number_of_annotations = len(glob.glob("{}/*xml".format(abs_annotations_path)))

        try:
            assert (number_of_images == number_of_annotations)
        except AssertionError:
            print("{} has {} images and \n".format(abs_images_path, number_of_images) +
                  "{} has {} annotations".format(abs_annotations_path, number_of_annotations))

    def partition_data_create_list(self):
        proj_path = os.path.join(self.config.ABSOLUTE_OUTPUT_PROJECT_PATH, self.config.PROJECT_NAME)
        abs_images_path = os.path.join(self.config.ABSOLUTE_DATASETS_PATH,
                                       self.config.DATASET_FODLER,
                                       self.config.DATASET_IDENTIFIER,
                                       self.config.IMAGE_FOLDER_NAME)

        data_partition_file_dir = os.path.join(self.config.ABSOLUTE_DATASETS_PATH,
                                               self.config.DATASET_FODLER,
                                               self.config.DATASET_IDENTIFIER,
                                               "ImageSets",
                                               "Main")

        # todo fix this to handle the images properly
        files = glob.glob("{}/*{}".format(abs_images_path, self.config.IMAGE_EXTENSION))
        try:
            os.makedirs(data_partition_file_dir)
        except FileExistsError:
            print(data_partition_file_dir, " exists")

        # create files if do not exist
        trainvaltxt_file = "{}/trainval.txt".format(data_partition_file_dir)
        trainvaloutput_file = "{}/trainval.txt".format(proj_path)
        testtxt_file = "{}/test.txt".format(data_partition_file_dir)
        testoutput_file = "{}/test.txt".format(proj_path)
        if os.path.exists(trainvaltxt_file):
            os.remove(trainvaltxt_file)
        if os.path.exists(testtxt_file):
            os.remove(testtxt_file)

        random.seed(7789)  # seed for the random numbers
        with io.open(trainvaloutput_file, 'w', newline='\n') as trainvalwriteproj, \
                io.open(trainvaltxt_file, 'w', newline='\n') as trainvalwrite, \
                io.open(testoutput_file, 'w', newline='\n') as testwriteproj, \
                io.open(testtxt_file, 'w', newline='\n') as testwrite:
            for images in files:
                filename = os.path.splitext(os.path.basename(images))[0]
                write_me = "{}\n".format(filename)
                image_path = os.path.join(self.config.IMAGE_FOLDER_NAME, "{}.{}".format(filename, self.config.IMAGE_EXTENSION))
                anno_path = os.path.join(self.config.ANNOTAION_FOLDERNAME, "{}.xml".format(filename))
                proj_write_me = "{} {}\n".format(image_path, anno_path)
                if random.randint(0, 100) < self.config.TEST_SET_PERCENTAGE:
                    testwrite.writelines(write_me)
                    testwriteproj.writelines(proj_write_me)
                else:
                    trainvalwrite.writelines(write_me)
                    trainvalwriteproj.writelines(proj_write_me)
        return testoutput_file, trainvaloutput_file

    @staticmethod
    def shuffle_file(the_file):
        with open(the_file, 'r') as source:
            data = [(random.random(), line) for line in source]
        data.sort()
        with open(the_file, 'w') as target:
            for _, line in data:
                target.write(line)

    @staticmethod
    def run(some_executable, some_args: list):
        some_command = [some_executable]
        some_command.extend(some_args)
        print(some_command)
        subprocess.call(some_command)

    def handle_single_prototxt_file(self, input_file, output_file):
        replace_me_1 = "cls1x"
        replace_me_3 = "cls3x"
        replace_me_6 = "cls6x"

        classes = len(self.config.TEST_CLASSES) + 1

        replacement_1 = str(classes)
        replacement_3 = str(classes * 3)
        replacement_6 = str(classes * 6)
        with io.open(input_file, 'r') as fin, io.open(output_file, 'w') as fout:
            for line in fin:
                add_line = line.replace(replace_me_1, replacement_1)
                add_line = add_line.replace(replace_me_3, replacement_3)
                add_line = add_line.replace(replace_me_6, replacement_6)
                fout.write(add_line)

    def process_prototxt_files(self):
        proj_path = os.path.join(self.config.ABSOLUTE_OUTPUT_PROJECT_PATH, self.config.PROJECT_NAME)
        output_trainfile = os.path.join(proj_path, "MobileNetSSD_train.prototxt")
        output_testfile = os.path.join(proj_path, "MobileNetSSD_test.prototxt")
        # output_deploybnfile = os.path.join(proj_path, "MobileNetSSD_deploy_bn.prototxt")
        output_deployfile = os.path.join(proj_path, "MobileNetSSD_deploy.prototxt")

        # copy from ABSOLUTE_PATH_NETWORK to output project path
        # template files
        tempate_trainfile = os.path.join(self.config.ABSOLUTE_PATH_NETWORK, "template", "MobileNetSSD_train_template.prototxt")
        tempate_testfile = os.path.join(self.config.ABSOLUTE_PATH_NETWORK, "template", "MobileNetSSD_test_template.prototxt")
        # tempate_deploybnfile = os.path.join(self.config.ABSOLUTE_PATH_NETWORK, "template", "MobileNetSSD_deploy_bn_template.prototxt")
        tempate_deployfile = os.path.join(self.config.ABSOLUTE_PATH_NETWORK, "template", "MobileNetSSD_deploy_template.prototxt")

        self.handle_single_prototxt_file(tempate_trainfile, output_trainfile) # todo add the change the batch size
        self.handle_single_prototxt_file(tempate_testfile, output_testfile)
        # self.handle_single_prototxt_file(tempate_deploybnfile, output_deploybnfile)
        self.handle_single_prototxt_file(tempate_deployfile, output_deployfile)

    def copy_additional_prototxt_files(self):
        proj_path = os.path.join(self.config.ABSOLUTE_OUTPUT_PROJECT_PATH, self.config.PROJECT_NAME)
        prototxtglob = glob.glob("{}/*prototxt".format(self.config.ABSOLUTE_PATH_NETWORK))
        for each in prototxtglob:
            bname = os.path.basename(each)
            outpath = os.path.join(proj_path, bname)
            shutil.copy2(each, outpath)

    def create_annoset(self):
        exe = self.config.PATH_TO_CONVERT_ANNOSET_DOT_EXE
        test_args = self.get_annoset_args("test")
        trainval_args = self.get_annoset_args("trainval")
        self.run(exe, test_args)
        self.run(exe, trainval_args)

    def get_annoset_args(self, testtrainval):
        proj_path = os.path.join(self.config.ABSOLUTE_OUTPUT_PROJECT_PATH, self.config.PROJECT_NAME)
        labelmap_prototxt = os.path.join(proj_path, "labelmap.prototxt")
        testtranval_txt = os.path.join(proj_path, "{}.txt".format(testtrainval))
        testtranval_lmdb = os.path.join(proj_path, "{}_lmdb".format(testtrainval))
        dataset_folder = os.path.join(self.config.ABSOLUTE_DATASETS_PATH,
                                      self.config.DATASET_FODLER,
                                      self.config.DATASET_IDENTIFIER, "")
        # todo might not want everything hardcoded
        return ["--anno_type=detection",
                "--label_map_file={}".format(labelmap_prototxt),
                "--label_type=xml",
                "--check_label=True",
                "--min_dim=0",
                "--max_dim=0",
                "--resize_height=0",
                "--resize_width=0",
                "--backend=lmdb",
                "--shuffle=False",
                "--check_size=False",
                "--encode_type={}".format(config.IMAGE_EXTENSION),
                "--encoded=True",
                "--gray=False",
                "{}".format(dataset_folder),
                "{}".format(testtranval_txt),
                "{}".format(testtranval_lmdb)]

    def make_snapshot_folder(self):
        proj_path = os.path.join(self.config.ABSOLUTE_OUTPUT_PROJECT_PATH, self.config.PROJECT_NAME, "snapshot")
        path_to_create = "{}".format(proj_path)
        os.makedirs(path_to_create, exist_ok=True)

    def create_label_file(self):
        labelfile = os.path.join(self.config.ABSOLUTE_OUTPUT_PROJECT_PATH, self.config.PROJECT_NAME, "labels.txt")
        with_background = ["background"]
        with_background.extend(self.config.TEST_CLASSES)
        with io.open(labelfile, 'w', newline='\n') as fpwrite:
            for each_class in with_background:
                fpwrite.write("{}\n".format(each_class))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # try:
        input_config = sys.argv[1]
        config = Config.read_json_to_config(input_config)
        # fix the items in the config
        config.get_classes_from_annotations()
        proj_path = os.path.join(config.ABSOLUTE_OUTPUT_PROJECT_PATH, config.PROJECT_NAME)
        if ope(proj_path):
            print("============= {} EXISTS =============".format(proj_path))
            print("Remove rename the project or remove {}".format(proj_path))
            raise Exception("{} EXISTS".format(proj_path))
        else:
            print("============= {} DNE =============".format(proj_path))

        prepare_for_training = Pft(config)
        prepare_for_training.generate_labelmap_prototxt_file()
        test_file, trainval_file = prepare_for_training.partition_data_create_list()
        prepare_for_training.shuffle_file(trainval_file)
        abs_dataset_location = os.path.join(config.ABSOLUTE_DATASETS_PATH,
                                            config.DATASET_FODLER,
                                            config.DATASET_IDENTIFIER)

        test_name_size = os.path.join(proj_path, "test_name_size.txt")
        some_arguments = [abs_dataset_location, test_file, test_name_size]
        # create image size txt
        prepare_for_training.run(config.PATH_TO_GET_IMAGE_SIZE_DOT_EXE, some_arguments)
        prepare_for_training.process_prototxt_files()
        prepare_for_training.copy_additional_prototxt_files()
        prepare_for_training.create_annoset()
        prepare_for_training.make_snapshot_folder()
        prepare_for_training.create_label_file()

        outuput_config = os.path.join(proj_path, "{}.config".format(config.PROJECT_NAME))
        shutil.copy2(input_config, outuput_config)
        output_training_file = os.path.join(proj_path, "train.py")
        output_testing_file = os.path.join(proj_path, "test.py")
        output_config_py_file = os.path.join(proj_path, "Config.py")

        shutil.copy2("train.py", output_training_file)
        shutil.copy2("test.py", output_testing_file)
        shutil.copy2("Config.py", output_config_py_file)

    else:
        print("Usage: {} <config_file>")
        pass
