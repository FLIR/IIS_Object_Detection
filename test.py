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
    snapshot_glob = glob.glob("snapshot/*caffemodel")
    if len(snapshot_glob) < 1:
        return
    else:
        snapshot_glob.sort(key=os.path.getmtime)
        return snapshot_glob[-1]


def test_latest_snapshot(config: Config):
    executable = config.PATH_TO_CAFFE_DOT_EXE
    execution_args = ["train", "-solver=solver_test.prototxt", "-weights={}".format(get_latest_snapshot())]
    run(executable, execution_args)
    pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_config = sys.argv[1]
        config = Config.read_json_to_config(input_config)
        test_latest_snapshot(config)

    else:
        print("Usage: {} <config_file>")
        pass
