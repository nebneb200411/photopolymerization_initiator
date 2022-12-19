import os

class ExtensionChanger:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def to_log(self):
        root, ext = os.path.splitext(self.file_path)
        file_path = self.file_path.replace(ext, '.log')
        return file_path

    def replacer(self, extension):
        root, ext = os.path.splitext(self.file_path)
        file_path = self.file_path.replace(ext, extension)
        return file_path

def add_log(path, neutral=True):
    if neutral:
        path = path + '_n.log'
    else:
        path = path + '_rc.log'
    return path