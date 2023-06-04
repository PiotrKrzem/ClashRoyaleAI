from typing import List, Dict
from card import Card

FILTER_CARD_ID = lambda card, id: card.original_id == id

# Clash royale deck
class Deck():
    def __init__(self, card_ids: List, cards: List[Card], rank: int):
        '''
        Method initialized clash royale deck.

        Parameters:
        card_ids - original identifiers of cards in the deck
        cards    - all cards present in the game
        rank     - ranking of a given deck
        '''
        self.rank = rank
        self.cards = [next(filter(lambda card: FILTER_CARD_ID(card, id), cards)) for id in card_ids]


def filter_duplicated_decks(decks: List[List[int]]) -> Dict[int, List[int]]:
    '''
    Method filters out duplicated decks.
    If duplicate occurred, deck with higher rank is taken into consideration.

    Parameters:
    decks - list of decks (each deck consists of card indexes)
    '''
    final_decks: dict[int, List[int]] = dict()

    for idx, deck in enumerate(decks):
        if deck not in final_decks.values():
            final_decks[idx] = deck
    
    return final_decks