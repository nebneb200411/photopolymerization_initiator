import os
import subprocess
from dotenv import load_dotenv
load_dotenv()


def run_mopac(dats_path):
    subprocess.call('{} "{}"'.format(os.environ['MOPAC_PATH'], dats_path), shell=True)