import os
import glob
import logging
import subprocess
from tqdm import tqdm

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


error_logger = setup_logger("error", "errors.log", logging.ERROR)
info_logger = setup_logger("info", "info.log", logging.INFO)

avis = glob.glob("Dataset/**/**/**/*.avi")


for avi in tqdm(avis):
    folder_path = os.path.dirname(avi)
    abs_path = os.path.abspath(avi)
    filename = os.path.basename(avi)
    folder_abs_path = os.path.abspath(folder_path)

    try:
        for img in os.listdir(folder_abs_path):
            # print(img)
            if img.endswith(".jpg") or img.endswith(".png"):
                os.remove(os.path.join(folder_abs_path, img))
            if os.path.isdir(os.path.join(folder_abs_path, img)) and img.endswith("frames"):
                os.rmdir(os.path.join(folder_abs_path, img))
                # print(os.path.join(folder_abs_path, img))

        output = subprocess.run('ffmpeg -i "' + abs_path + '" -q:v 2 ' + filename[:-4] + '%04d.png -hide_banner',
                                shell=True, cwd=folder_abs_path, check=True, capture_output=True)
    except Exception as e:
        error_logger.error("Skipped "+abs_path+'/. Error: '+str(e))
    else:
        info_logger.info("Ouput "+str(output))

print("================================================================================\n")
print("Frame Extraction Successful")
