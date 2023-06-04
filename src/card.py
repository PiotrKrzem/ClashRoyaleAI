from typing import Any
from data.data_parser import import_cards_data

FILTER_CARD_KEY = lambda card, name: card['key'] == name

# Clash royale card with its statistics data
class Card():
    def __init__(self, card_stats: Any, id: int):
        '''
        Method initialized clash royale card.

        Parameters:
        card_stats - card statistic data
        id         - identifier used in model training
        '''
        self.id = id
        self.name = card_stats['data-card']
        self.win_rate = float(card_stats['data-winpercent'])
        self.usage = float(card_stats['data-usage']) * 100
        self.fill_card_data()

    def fill_card_data(self):
        '''
        Method completes the card statistic data with its characteristic information.
        '''
        cards_data = import_cards_data()
        card = next(filter(lambda card: FILTER_CARD_KEY(card, self.name), cards_data))

        self.original_id = card['id']
        self.type = card['type']
        self.cost = card['elixir']
        self.rarity = card['rarity']