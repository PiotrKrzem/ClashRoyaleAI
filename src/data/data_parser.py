import json

from os import path
from enum import Enum

def get_file_path(file_name: str) -> str:
    '''
    Method returns the full path to the file with data.

    Parameters:
    file_name - name of the file for which the path is to be returned
    '''
    file_absolute_path = path.dirname(__file__)
    return path.join(file_absolute_path, file_name)


def import_cards_data():
    '''
    Method returns data of cards
    '''
    file = open(get_file_path('cards.json'))
    data = json.load(file)
    file.close()

    return data

class ClashRoyaleData(Enum):
    CARDS = import_cards_data()