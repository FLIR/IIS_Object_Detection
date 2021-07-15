import io
import random
import subprocess
import shutil
import sys
from Config import *


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

class And(object):

    def __init__(self, config: Config):
        super(And, self).__init__()
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
        label_map_prototxt_file = "labelmap.prototxt"

        dump_string = ""
        dump_string += background_class
        class_index = 0
        for each_class in self.config.TEST_CLASSES:
            class_index += 1
            dump_string += self.other_class(each_class, class_index, each_class)

        with io.open(label_map_prototxt_file, 'w', newline='\n') as fpwrite:
            fpwrite.write(dump_string)

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
        # get this again
        labelmap_prototxt = os.path.join(proj_path, "labelmap.prototxt")
        # append to this file
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
                "--encode_type={}".format(self.config.IMAGE_EXTENSION),
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

    @staticmethod
    def rename_old_image_list_files():
        number = len(glob.glob("test*txt"))

        old_test = "test{}.txt".format(number)
        old_trainval = "test{}.txt".format(number)

        os.rename("test.txt", old_test)
        os.rename("trainval.txt", old_trainval)
        return old_test, old_trainval

    @staticmethod
    def remove_lmdb_files():
        os.rmdir("test_lmdb")
        os.rmdir("trainval_lmdb")

    @staticmethod
    def merge_two_lists_remove_dupes(new_classes, old_classes):
        in_new = set(new_classes)
        in_old = set(old_classes)

        in_new_but_not_in_old = in_new - in_old

        all_classes = old_classes + list(in_new_but_not_in_old)

        return all_classes

    @staticmethod
    def get_additional_information(old_config, new_config):
        old_dataset_path = old_config.get_dataset_path()
        new_dataset_path = new_config.get_dataset_path()

        odp = splitall(old_dataset_path)
        ndp = splitall(new_dataset_path)

        minlen = min(len(odp), len(ndp))

        same = []
        weron = 0
        for x in range(minlen):
            if odp[x] == ndp[x]:
                same.append(odp[x])
            else:
                weron = x
                break

        output_concatinated = ""

        for each in same:
            output_concatinated = os.path.join(output_concatinated, each)

        old_dataset_path_out = ""
        for x in range(weron, len(odp), 1):
            print(odp[x])
            old_dataset_path_out = os.path.join(old_dataset_path_out, odp[x])

        new_dataset_path_out = ""
        for x in range(weron, len(ndp), 1):
            new_dataset_path_out = os.path.join(new_dataset_path_out, ndp[x])

        return output_concatinated, old_dataset_path_out, new_dataset_path_out

    @staticmethod
    def prepend_to_each_line(file, string, destination):
        with io.open(file, 'r') as source, io.open(destination, 'w',  newline='\n') as dest:
            for line in source:
                image, annotation = line.split(' ')
                dest.write("{}{} {}{}".format(string,image,string,annotation))

    @staticmethod
    def merge_two_files(file1, file2, outfile):
        with io.open(file1, 'r') as f1, io.open(file2, 'r') as f2, io.open(outfile, 'w') as out:
            for line in f1:
                out.write(line)
            for line in f2:
                out.write(line)

    def update_lists(self, new_file, old_file, prepend_to_new, prepend_to_old):
        temp_new_file = "tmpn"
        self.prepend_to_each_line(new_file, prepend_to_new, temp_new_file)
        temp_old_file = "tmpo"
        self.prepend_to_each_line(old_file, prepend_to_old, temp_old_file)
        self.merge_two_files(temp_new_file, temp_old_file, new_file)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        # try:
        config_1 = sys.argv[1]
        config_2 = sys.argv[2]
        new_config = Config.read_json_to_config(config_1)
        old_config = Config.read_json_to_config(config_2)
        # fix the items in the config
        new_config.get_classes_from_annotations() # gets the new annotaions
        old_config.get_classes_from_annotations()

        new_classes = new_config.TEST_CLASSES
        old_classes = old_config.TEST_CLASSES

        # the project path will the current path as we're updating the project...

        appendnewdata = And(new_config)
        appendnewdata.config.TEST_CLASSES = appendnewdata.merge_two_lists_remove_dupes(new_classes, old_classes)
        # doublecheck where this guy goes
        appendnewdata.generate_labelmap_prototxt_file()
        # this part gives you a new list...

        old_test_file, old_trainval_file = appendnewdata.rename_old_image_list_files()
        new_test_file, new_trainval_file = appendnewdata.partition_data_create_list()

        appendnewdata.shuffle_file(new_trainval_file)

        output_concatinated, old_dataset_path_out, new_dataset_path_out = appendnewdata.get_additional_information(old_config, new_config)

        # this
        appendnewdata.remove_lmdb_files()
        appendnewdata.update_lists(new_test_file, old_test_file, new_dataset_path_out, old_dataset_path_out)
        appendnewdata.update_lists(new_trainval_file, old_trainval_file, new_dataset_path_out, old_dataset_path_out)
        appendnewdata.create_annoset()

        appendnewdata.create_label_file()
    else:
        print("Usage: {} <new_config_file> <old_config_file>")
        pass
