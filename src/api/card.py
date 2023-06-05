import numpy as np

from typing import Any, Dict
from src.data.data_parser import import_cards_data
from src.constants import *

FILTER_CARD_KEY = lambda card, name: card['key'] == name

TYPE_ID_MAP = {
    "Troop": 0,
    "Spell": 1,
    "Building": 2
}

RARITY_ID_MAP = {
    "Common": 0,
    "Rare": 1,
    "Epic": 2,
    "Legendary": 3,
    "Champion": 4
}

# Clash royale card with its statistics data
class Card():
    def __init__(self, card: Dict):
        '''
        Method initialized clash royale card.

        Parameters:
        card - card dictionary
        '''
        self.id = card['id']
        self.name = card['name']
        self.win_rate = card['win_rate']
        self.usage = card['usage']
        self.original_id = card['original_id']
        self.type = card['type']
        self.cost = card['cost']
        self.rarity = card['rarity']

    def to_vec(self):
        card_tensor = np.empty((CARD_ENCODING_SIZE), dtype=np.float32)
        card_tensor[0] = self.win_rate/100
        card_tensor[1] = self.usage/100
        card_tensor[2] = self.usage/10
        card_tensor[3 + TYPE_ID_MAP[self.type]] = 1
        card_tensor[6 + RARITY_ID_MAP[self.rarity]] = 1

        return card_tensor

    @classmethod
    def from_card_statistics(cls, card_stats: Any, id: int):
        '''
        Method initialized clash royale card.

        Parameters:
        card_stats - card statistic data
        id         - identifier used in model training
        '''
        card = dict()

        # filling information about card statistics
        card['id'] = id
        card['name'] = card_stats['data-card']
        card['win_rate'] = float(card_stats['data-winpercent'])
        card['usage'] = float(card_stats['data-usage']) * 100
        
        cards_data = import_cards_data()
        card_original = next(filter(lambda c: FILTER_CARD_KEY(c, card['name']), cards_data))

        # filling information about card characteristics
        card['original_id'] = card_original['id']
        card['type'] = card_original['type']
        card['cost'] = card_original['elixir']
        card['rarity'] = card_original['rarity']

        return cls(card)

