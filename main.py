import torch
import torch.nn.functional as F

from src.clash_royal_api import ClashRoyaleAPI

NR_OF_CARDS = 10
NR_OF_BUCKETS = 1
NR_OF_CARDS_PER_DECK = 4

ENCODER_DIM = 3
DENSE_NEURONS = 16
LEARNING_RATE = 1e-4
EPOCHS = 1000

class ClashRoyaleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nr_of_cards = NR_OF_CARDS
        self.nr_of_ranking_buckes = NR_OF_BUCKETS
        self.nr_of_cards_per_deck = NR_OF_CARDS_PER_DECK
        self.nr_of_neurons_per_dense = DENSE_NEURONS

        self.input_size = self.nr_of_cards + self.nr_of_ranking_buckes
        self.encoding_size = ENCODER_DIM

        self.encoding_layer = torch.nn.Linear(
            in_features=self.input_size, 
            out_features=self.encoding_size,
            bias=False
        )
        self.dense_layer_1 = torch.nn.Linear(
            in_features=self.encoding_size,
            out_features=self.nr_of_neurons_per_dense,
            bias=True
        )
        self.dense_layer_2 = torch.nn.Linear(
            in_features=self.nr_of_neurons_per_dense,
            out_features=self.nr_of_neurons_per_dense,
            bias=True
        )
        self.classifier_layer = torch.nn.Linear(
            in_features=self.nr_of_neurons_per_dense,
            out_features=self.input_size,
            bias=True
        )

        # self.missing_card_recovery_loss = torch.nn.CrossEntropyLoss()
        # self.deck_similarity_to_ranking_loss = torch.nn.BCEWithLogitsLoss()
        # self.deck_similarity_to_ranking_loss = torch.nn.MSELoss()
        # self.deck_similarity_to_ranking_loss = lambda x, y: (x - torch.min(x) - (y - torch.min(y))).sum()

    def forward(self, cards: torch.Tensor, ranking: torch.Tensor):
        """
        Inputs:
            cards - Tensor of shape (b, nr_of_cards_per_deck), where last dimension is a list of int indices of cards in the deck
            ranking - Tensor of shape (b, 1), where last dimension is the ranking number, int in range [0,NR_OF_BUCKETS)
        """
        batch_size = cards.shape[0]

        target_cards_one_hot: torch.Tensor = F.one_hot(cards, num_classes = self.input_size)
        # -> (batch_size, nr_of_cards_per_deck, input_size)
        target_cards_one_hot = target_cards_one_hot.view([batch_size * self.nr_of_cards_per_deck, self.input_size]).float()
        # -> (batch_size * nr_of_cards_per_deck, input_size)
        target_cards_encoding = self.encoding_layer(target_cards_one_hot)
        # -> (batch_size * nr_of_cards_per_deck, encoding_size)

        deck_one_hot = torch.zeros((batch_size, self.input_size), dtype=torch.float32)
        # -> (batch_size, input_size)
        deck_one_hot[:, cards] = 1.0
        # -> (batch_size, input_size)
        deck_encoding: torch.Tensor = self.encoding_layer(deck_one_hot)
        # -> (batch_size, encoding_size)
        deck_encoding = deck_encoding.repeat(self.nr_of_cards_per_deck, 1).view((batch_size * self.nr_of_cards_per_deck, self.encoding_size))
        # -> (batch_size * nr_of_cards_per_deck, encoding_size)

        offset_ranking = ranking.add(self.nr_of_cards)
        # -> (batch_size, 1)
        ranking_one_hot: torch.Tensor = F.one_hot(offset_ranking, num_classes = self.input_size)
        # -> (batch_size, 1, input_size)
        ranking_one_hot = ranking_one_hot.view((batch_size, self.input_size)).float()
        # -> (batch_size, input_size)
        ranking_encoding: torch.Tensor = self.encoding_layer(ranking_one_hot)
        # -> (batch_size, encoding_size)
        ranking_encoding = ranking_encoding.repeat(self.nr_of_cards_per_deck, 1).view((batch_size * self.nr_of_cards_per_deck, self.encoding_size))
        # -> (batch_size * nr_of_cards_per_deck, encoding_size)

        # ========================================================================
        deck_with_cards_removed = deck_encoding.subtract(target_cards_encoding)
        # -> (batch_size * nr_of_cards_per_deck, encoding_size)
        predicted_removed_card = self.dense_layer_1(deck_with_cards_removed)
        predicted_removed_card = F.relu(predicted_removed_card)
        # -> (batch_size * nr_of_cards_per_deck, nr_of_neurons_per_dense)
        predicted_removed_card = self.dense_layer_2(predicted_removed_card)
        predicted_removed_card = F.relu(predicted_removed_card)
        # -> (batch_size * nr_of_cards_per_deck, nr_of_neurons_per_dense)
        predicted_removed_card = self.classifier_layer(predicted_removed_card)
        predicted_removed_card = F.softmax(predicted_removed_card, dim = 1)
        # -> (batch_size * nr_of_cards_per_deck, input_size)

        predicted_removed_card = self.encoding_layer(predicted_removed_card)
        rebuilt_decks = deck_with_cards_removed.add(predicted_removed_card)
        rebuilt_deck_encoding = torch.mean(rebuilt_decks, dim=0)

        return rebuilt_deck_encoding, ranking_encoding
    
    # def compute_loss(self, rebuilt_deck_encoding, ranking_encoding):
    #     loss = self.deck_similarity_to_ranking_loss(rebuilt_deck_encoding, ranking_encoding)
    #     return loss

    def compute_loss(self, rebuilt_deck_encoding, ranking_encoding):
        rebuilt_deck_encoding = rebuilt_deck_encoding - torch.min(rebuilt_deck_encoding)
        ranking_encoding = ranking_encoding - torch.min(ranking_encoding)
        loss = torch.abs((rebuilt_deck_encoding - ranking_encoding).sum())
        return loss

    def get_encoding(self, card_idx):
        card_encoding: torch.Tensor = self.encoding_layer.weight[:, card_idx]
        return card_encoding.detach().numpy()

batch_size = 1
cards = torch.randint(0, NR_OF_CARDS, (batch_size, NR_OF_CARDS_PER_DECK))
rankings = torch.randint(0, NR_OF_BUCKETS, (batch_size, 1))

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CLASH ROYALE API :) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
api = ClashRoyaleAPI()
decks = api.get_top_players_decks()

model = ClashRoyaleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss = 0
for t in range(EPOCHS):
    # Forward pass
    rebuilt_deck, ranking_encoding = model(cards, rankings)

    # Compute and print loss
    loss = model.compute_loss(rebuilt_deck, ranking_encoding)

    if t % 10 == 9:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {loss}')