import subprocess
import os
def run_gaussian(file_path):
    subprocess.run('g16 "{}"'.format(file_path), shell=True)

def chk_to_fchk(chk_filepath):
    filename, _ = os.path.splitext(chk_filepath)
    fchk_filepath = filename + '.fchk'
    subprocess.run('formchk {} {}'.format(chk_filepath, fchk_filepath), shell=True)