import numpy as np
from typing import List, Dict
from src.api.card import Card
from src.constants import *

FILTER_CARD_ID = lambda card, id: card.original_id == id

# Clash royale deck
class Deck():
    def __init__(self, cards: List[Card], rank: int):
        '''
        Method initialized clash royale deck.

        Parameters:
        cards    - cards of a given deck
        rank     - ranking of a given deck
        '''
        self.cards = [Card(card if type(card) is dict else card.__dict__) for card in cards]
        self.cards.sort(key=lambda card:card.id)
        self.rank = rank

    def cards_to_vec(self):
        cards_tensor = np.empty((NR_OF_CARDS_PER_DECK, CARD_ENCODING_SIZE))
        for i, card in enumerate(self.cards):
            cards_tensor[i] = card.to_vec()
        return cards_tensor

    def split(self, adj_matrix):
        removed_idx = np.random.randint(0, NR_OF_CARDS)

        cards_remaining_encoding = np.zeros((1, CARD_ENCODING_SIZE), dtype=np.float32)
        card_removed_encoding = np.zeros((1, NR_OF_CARDS+NR_OF_BUCKETS), dtype=np.float32)
        for i in range(NR_OF_CARDS_PER_DECK):
            if i == removed_idx:
                card_removed_encoding[0, self.cards[i].id] = 1
            else:
                cards_remaining_encoding[0] += self.cards[i].to_vec() * adj_matrix[removed_idx][i]
        return cards_remaining_encoding, card_removed_encoding

    def one_hot_deck(self):
        one_hot_encoded_deck = np.zeros((1, NR_OF_CARDS+NR_OF_BUCKETS), dtype=np.float32)
        one_hot_encoded_deck[0, list(map(lambda card:int(card.id), self.cards))] = 1
        return one_hot_encoded_deck

    def one_hot_ranking(self):
        one_hot_encoded_ranking = np.zeros((1, NR_OF_CARDS+NR_OF_BUCKETS), dtype=np.float32)
        one_hot_encoded_ranking[0, NR_OF_CARDS + self.rank//(MAX_RANKING//NR_OF_BUCKETS)] = 1
        return one_hot_encoded_ranking

    @classmethod
    def from_card_ids(cls, card_ids: List, cards: List[Card], rank: int):
        '''
        Method initialized clash royale deck from card ids.

        Parameters:
        card_ids - original identifiers of cards in the deck
        cards    - all cards present in the game
        rank     - ranking of a given deck
        '''
        rank = rank
        cards = [next(filter(lambda card: FILTER_CARD_ID(card, id), cards)) for id in card_ids]

        return cls(cards, rank)


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