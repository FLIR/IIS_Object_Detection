from Config import *
import sys
import glob
import os
import subprocess


def run(some_executable, some_args: list):
    some_command = [some_executable]
    some_command.extend(some_args)
    print(some_command)
    subprocess.call(some_command)


def get_latest_snapshot():
    '''this is running from the project folder we can assume that we are ignoring absolute path'''
    snapshot_glob = glob.glob("snapshot/*solverstate")
    if len(snapshot_glob) < 1:
        return
    else:
        snapshot_glob.sort(key=os.path.getmtime)
        return snapshot_glob[-1]


def start_training(config: Config):
    ## {caffe_exe} train -solver=solver_train.prototxt -weights={pretrained}
    # sets the config's continue flag to true
    config.CONTINUE_TRAINING = True
    config.CAFFE_EXECUTION_COUNT += 1
    config.dump_json_to_file("{}-{}.config".format(config.PROJECT_NAME, config.CAFFE_EXECUTION_COUNT))
    executable = config.PATH_TO_CAFFE_DOT_EXE
    pretrained_model = os.path.join(config.ABSOLUTE_PATH_NETWORK, config.PRETRAINED_NETWORK_FILE)
    config.update_max_iterations()
    config.update_batch_size()
    execution_args = ["train", "-solver=solver_train.prototxt", "-weights={}".format(pretrained_model)]
    run(executable, execution_args)


def resume_training(config: Config):
    ## {caffe_exe} train -solver=solver_train.prototxt -snapshot={latest_snapshot}
    config.CAFFE_EXECUTION_COUNT += 1
    config.dump_json_to_file("{}-{}.config".format(config.PROJECT_NAME, config.CAFFE_EXECUTION_COUNT))
    executable = config.PATH_TO_CAFFE_DOT_EXE
    execution_args = ["train", "-solver=solver_train.prototxt", "-snapshot={}".format(get_latest_snapshot())]
    run(executable, execution_args)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_config = sys.argv[1]
        config = Config.read_json_to_config(input_config)
        if config.CONTINUE_TRAINING:
            resume_training(config)
        else:
            start_training(config)

    else:
        config = Config.get_latest_config()
        if config.CONTINUE_TRAINING:
            resume_training(config)
        else:
            start_training(config)
