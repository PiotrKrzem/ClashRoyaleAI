# ------------ API CLIENT CONSTANTS --------------------

API_URL = 'https://api.clashroyale.com/v1'
STATS_API_URL = 'https://royaleapi.com/cards/popular?time=7d&mode=grid&cat=TopRanked&sort=rating'

API_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6ImNmMGMyODU3LTNjOTItNGMzNS04YjljLWRlMmEzZjdmMWEzYSIsImlhdCI6MTY4NTgyNTU4NCwic3ViIjoiZGV2ZWxvcGVyL2VhZWM5NjE5LTYyODEtZmU0MC0zNTIzLWVlOTEzODhlNWE3NiIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyI3OS4xODQuMTY5LjIxIl0sInR5cGUiOiJjbGllbnQifV19.YTKh5WQFphB87hEKSafqMbiIGuXyvTc6Y70AwXdMcr_mJHcOXlT0QgyCg0CA8HAsvVUSWTU4Q8khzCheq7WBlQ'

# ----------- API REQUEST CONSTANTS ---------------------

SEASON = '2023-03'
MAX_PLAYERS = 1000
MAX_CARDS = 100

# ------------------------------------------------------

NR_OF_CARDS = 109
NR_OF_BUCKETS = 10
NR_OF_CARDS_PER_DECK = 8
CARD_ENCODING_SIZE = 11 # wr, use, type (tr/sp/bu), cost, rarity (comm, rare, epic, leg, ch)
MAX_RANKING = 1000

ENCODER_DIM = 8
DENSE_NEURONS = 128
LEARNING_RATE = 1e-4

EPOCHS = 100
