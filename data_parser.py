from enum import Enum
from typing import Any
from requests import get
from data.api_constants import API_TOKEN, API_URL, MAX_PLAYERS, MAX_CARDS, SEASON

GET_PLAYER_ID = lambda player: player['tag'].replace('#', '%23')



def handle_top_players(response: Any):
    players = response.json()['items'][:MAX_PLAYERS]

    return dict((GET_PLAYER_ID(player), idx) for idx, player in enumerate(players))

def handle_player_cards(response: Any):
    battles = response.json()
    legendBattles = [b for b in battles if b['type'] == 'pathOfLegend']

    return [card['id'] for card in legendBattles[0]['team'][0]['cards'][:MAX_CARDS]]


class ApiRequests(Enum):
    GET_TOP_PLAYERS = {
        'url': lambda arg: f'/locations/global/pathoflegend/{arg}/rankings/players',
        'handler': handle_top_players
    }
    GET_PLAYER_CARDS = {
        'url': lambda arg: f'/players/{arg}/battlelog',
        'handler': handle_player_cards
    }



class ClashRoyaleAPI():
    def __init__(self, token: str = API_TOKEN):
        self.authorization = { 'Authorization': f'Bearer {token}' }

    def get_top_players(self):
        players: dict = self.handle_request(ApiRequests.GET_TOP_PLAYERS, SEASON)
        print(self.handle_request(ApiRequests.GET_PLAYER_CARDS, list(players.keys())[0]))


    def handle_request(self, request: ApiRequests, arg: str):
        response = get(f'{API_URL}{request.value["url"](arg)}', headers=self.authorization)

        if response.status_code == 200:
            return request.value["handler"](response)
        else:
            print('Error while retrieving the data:')
            print(response.json()) 


api = ClashRoyaleAPI()
api.get_top_players()