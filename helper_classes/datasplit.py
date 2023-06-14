from enum import Enum, auto
from typing import List


class DataSplit(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()

    def __str__(self):
        return self.name

    @staticmethod
    def get_all_splits() -> List:
        return [DataSplit.TRAIN, DataSplit.VAL, DataSplit.TEST]


if __name__ == '__main__':
    a = DataSplit.TEST
    print(a)
    print(a.get_all_splits())
    for x in a.get_all_splits():
        print(x==DataSplit.VAL)
