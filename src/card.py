from typing import Any, Dict
from data.data_parser import import_cards_data

FILTER_CARD_KEY = lambda card, name: card['key'] == name

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