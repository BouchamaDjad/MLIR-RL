from collections import defaultdict
from typing import Any, Dict, List

def init_run(*args, **kwargs):
    return NeptuneMock(*args, **kwargs)

class fileObj:
    def __init__(self):
        self.files = []

    def upload_files(self, files):
        self.files.extend(files)

    def upload(self, file):
        self.files.append(file)

    def __str__(self):
        return str(self.files)
    def __repr__(self):
        return str(self.files)


class NeptuneMock:
    def __init__(self, *args, **kwargs):
        self.dictionary = defaultdict(list)

        self.dictionary["params"] = fileObj()
        self.dictionary["src"] = fileObj()
        self.dictionary["config"] = fileObj()

        self.args = args
        self.kwargs = kwargs

    def stop(self):
        pass

    def __dict__(self) -> Dict[str, List[Any] | fileObj]:
        return self.dictionary
    
    def __getitem__(self, key):  
        return self.dictionary[key]

    def __str__(self):
        return str(self.dictionary) + ': args = ' + str(self.args) + ': kwargs = ' + str(self.kwargs)
    
    def __repr__(self):
        return self.__str__()

# from unittest.mock import MagicMock

#a = MagicMock()

