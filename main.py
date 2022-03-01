import time
import traceback
from pathlib import Path

from src.ExecutionRun import ExecutionRun

# Common Paths and directory names:
CURRENT_CONFIG_FILE_MSG = """
###########################################################################
Current config file: {}
###########################################################################
"""

PATH_CONFIG_FILES = 'config_files'

if __name__ == '__main__':
    # TODO: Please add your configuration files here:
    config_list = [
        '50px_alexander_71pics_sphere_nerf.yaml',
    ]
    for config_filename in config_list:
        print(CURRENT_CONFIG_FILE_MSG.format(config_filename))
        main_execution = None
        try:
            start_main_time = time.time()
            main_execution = ExecutionRun(Path(PATH_CONFIG_FILES) / config_filename)
            main_execution.start()
            passed_time = time.time() - start_main_time
            print(f"Time elapsed for main function {passed_time :.3} seconds with config file {config_filename}.")
        except Exception as e:
            print("Caught an exception while running:", config_filename)
            print(Exception)
            print(traceback.format_exc())
            time.sleep(10)
            print("Exiting.")
            exit(1)
