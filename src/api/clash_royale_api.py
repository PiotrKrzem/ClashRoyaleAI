import urllib.request as rq
import numpy as np
import itertools as it
import json

from bs4 import BeautifulSoup
from enum import Enum
from typing import Any, List
from requests import get

from src.api.card import Card
from src.data.data_parser import get_file_path
from src.api.deck import Deck, filter_duplicated_decks
from src.constants import API_TOKEN, STATS_API_URL, API_URL, MAX_PLAYERS, MAX_CARDS, SEASON

# ----------------------- INLINE METHODS --------------------------------
GET_PLAYER_ID = lambda player: player['tag'].replace('#', '%23')
FILTER_BATTLE = lambda battle: battle['type'] == 'pathOfLegend'

# ---------------------- API REQUEST HANDLERS ---------------------------
def handle_top_players(response: Any):
    '''
    Method handles request to retrieve top players from the pathOfLegend league.

    ParameterS:
    reponse - response received to GET /rankings/players API request
    '''
    players = response.json()['items'][:MAX_PLAYERS]

    return dict((GET_PLAYER_ID(player), idx) for idx, player in enumerate(players))

def handle_player_cards(response: Any):
    '''
    Method handles request to get cards for a given player.
    Cards that are returned are taken from the first pathOfLegend battle.
    If the player has not played any battle then empty list is returned.

    ParameterS:
    reponse - response received to GET /battlelog API request
    '''
    battles = response.json()
    battle = next(filter(FILTER_BATTLE, battles), None)

    return [card['id'] for card in battle['team'][0]['cards'][:MAX_CARDS]] if battle else []


class ApiRequests(Enum):
    '''
    Enum that defines method for corresponding API requests.
    '''
    GET_TOP_PLAYERS = {
        'url': lambda arg: f'/locations/global/pathoflegend/{arg}/rankings/players',
        'handler': handle_top_players
    }
    GET_PLAYER_CARDS = {
        'url': lambda arg: f'/players/{arg}/battlelog',
        'handler': handle_player_cards
    }


# --------------------------- CLASH ROYALE API ----------------------------------
class ClashRoyaleAPI():
    def __init__(self, token: str = API_TOKEN):
        '''
        Method initializes clash royale api.

        Parameters:
        token - authorization token used to retrieve API data.
        '''
        self.authorization = { 'Authorization': f'Bearer {token}' }
        self.cards = self.get_cards_data()

    def get_top_players_decks(self):
        '''
        Method returns decks for top pathOfLeague players.
        '''
        players: dict = self.handle_request(ApiRequests.GET_TOP_PLAYERS, SEASON)
        decks = [self.handle_request(ApiRequests.GET_PLAYER_CARDS, player) for player in list(players.keys())]
        unique_decks = filter_duplicated_decks(decks)
        
        return [Deck.from_card_ids(card_ids, self.cards, idx) for idx, card_ids in unique_decks.items() if len(card_ids) > 0]

    def get_cards_data(self):
        '''
        Method returns cards data (statistical data + cards characteristics).
        '''
        # get webpage structure
        request_site = rq.Request(STATS_API_URL, headers={ "User-Agent" : "Mozilla/5.0" })
        api_cr = rq.urlopen(request_site)
        data = api_cr.read()
        api_cr.close()

        # read html tree
        soup = BeautifulSoup(data, features="lxml")

        # parse webpage to get ranking data
        div_container = soup.find("div", id = "page_content_container")
        div_content = div_container.find("div", id = "page_content")
        div_cards_container = div_content.find("div", { "class" : "ui container sidemargin0 popular_cards__container" })
        div_cards_segment = div_cards_container.find("div", { "class" : "ui attached segment" })
        div_cards_grid = div_cards_segment.find("div", { "class" : "card_grid__cards_container isotope_grid" })
        div_cards = div_cards_grid.find_all("div", { "class" : "grid_item" })

        return [Card.from_card_statistics(card, idx) for idx, card in enumerate(div_cards)]
    
    def store_player_decks(self, file_name: str):
        '''
        Method writes the players decks into a JSON file.

        Parameters:
        file_name - name of the file in which the data is to be stored
        '''
        decks = self.get_top_players_decks()
        json_str = json.dumps(decks, default = lambda x: x.__dict__)

        with open(get_file_path(file_name), 'w') as file:
            file.write(json_str)

    def read_player_decks(self, file_name: str):
        '''
        Method reads the players decks from a JSON file.

        Parameters:
        file_name - name of the file in which the data is to be stored
        '''
        with open(get_file_path(file_name), 'r') as file:
            file_data = file.read()
            json_data = json.loads(file_data)
            decks = [Deck(**deck) for deck in json_data]
            
            return decks
    
    def handle_request(self, request: ApiRequests, arg: str):
        '''
        Generic method used to handle API requests.

        Parameters:
        request - request sent to the API
        arg     - arguments passed in the request
        '''
        response = get(f'{API_URL}{request.value["url"](arg)}', headers=self.authorization)

        if response.status_code == 200:
            return request.value["handler"](response)
        else:
            print('Error while retrieving the data:')
            print(response.json()) 

    def get_adjacency_matrix_for_decks(self, decks: List[Deck]):
        '''
        Method returns an adjacency matrix based on given decks' cards.

        Parameters:
        decks - decks for which matrix is created
        '''
        matrix_size = len(self.cards)
        matrix = np.zeros((matrix_size, matrix_size))

        for deck in decks:
            cards = [card.id for card in deck.cards]
            card_pairs = list(it.combinations(cards, 2))

            for pair in card_pairs:
                matrix[pair[0], pair[1]] += 1
                matrix[pair[1], pair[0]] += 1

        print(matrix)
        return matrix

if __name__ == '__main__':
    api = ClashRoyaleAPI()
    decks = api.read_player_decks('big_cards_data.json')
    adj_matrix = api.get_adjacency_matrix_for_decks(decks)
    # api.store_player_decks('big_cards_data.json')