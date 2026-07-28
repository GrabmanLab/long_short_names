from urllib.request import urlretrieve
from zipfile import ZipFile
from shutil import rmtree
import os

TASK_DIR = "."


url = "https://osf.io/uqn93/?action=download&version=1"

filename = os.path.join(TASK_DIR, "encoding_images.zip")

if not os.path.exists(os.path.join(TASK_DIR, "encoding_images")):
    if not os.path.exists(os.path.join(url, filename)):
        urlretrieve(url, filename)

    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(TASK_DIR))

    os.remove(filename)