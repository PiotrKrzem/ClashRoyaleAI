import random
import torch
import math
import torch.nn.functional as F

NR_OF_CARDS = 100
NR_OF_BUCKETS = 10
NR_OF_CARDS_PER_DECK = 8
ENCODER_DIM = 3
DENSE_NEURONS = 128
LEARNING_RATE = 1e-8
EPOCHS = 100

class ClashRoyaleNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
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

        self.missing_card_recovery_loss = torch.nn.CrossEntropyLoss()
        self.deck_similarity_to_ranking_loss = torch.nn.BCEWithLogitsLoss()

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
        target_cards_softmax = target_cards_one_hot.softmax(dim=1)
        # -> (batch_size * nr_of_cards_per_deck, input_size)
        target_cards_encoding = self.encoding_layer(target_cards_one_hot)
        # -> (batch_size * nr_of_cards_per_deck, encoding_size)

        deck_one_hot = torch.zeros((batch_size, self.input_size), dtype=torch.float32)
        # -> (batch_size, input_size)
        deck_one_hot[cards] = 1.0
        # -> (batch_size, input_size)
        deck_encoding: torch.Tensor = self.encoding_layer(deck_one_hot)
        # -> (batch_size, encoding_size)
        deck_encoding = deck_encoding.repeat(1, self.nr_of_cards_per_deck).view((batch_size * self.nr_of_cards_per_deck, self.encoding_size))
        # -> (batch_size * nr_of_cards_per_deck, encoding_size)

        offset_ranking = ranking.add(self.nr_of_cards)
        # -> (batch_size, 1)
        ranking_one_hot: torch.Tensor = F.one_hot(offset_ranking, num_classes = self.input_size)
        # -> (batch_size, 1, input_size)
        ranking_one_hot = ranking_one_hot.view((batch_size, self.input_size)).float()
        # -> (batch_size, input_size)
        ranking_encoding = self.encoding_layer(ranking_one_hot)
        # -> (batch_size, encoding_size)
        ranking_encoding = ranking_encoding.repeat(1, self.nr_of_cards_per_deck).view((batch_size * self.nr_of_cards_per_deck, self.encoding_size))
        # -> (batch_size * nr_of_cards_per_deck, encoding_size)

        # ========================================================================
        deck_with_cards_removed = deck_encoding.subtract(target_cards_encoding)
        # -> (batch_size * nr_of_cards_per_deck, encoding_size)
        deck_with_cards_removed = self.dense_layer_1(deck_with_cards_removed)
        deck_with_cards_removed = F.tanh(deck_with_cards_removed)
        # -> (batch_size * nr_of_cards_per_deck, nr_of_neurons_per_dense)
        deck_with_cards_removed = self.dense_layer_2(deck_with_cards_removed)
        deck_with_cards_removed = F.tanh(deck_with_cards_removed)
        # -> (batch_size * nr_of_cards_per_deck, nr_of_neurons_per_dense)
        deck_with_cards_removed = self.classifier_layer(deck_with_cards_removed)
        deck_with_cards_removed_softmax = F.softmax(deck_with_cards_removed)
        # -> (batch_size * nr_of_cards_per_deck, input_size)

        return deck_with_cards_removed_softmax, target_cards_softmax, deck_encoding, ranking_encoding
    
    def compute_loss(self, deck_with_card_removed, missing_card, deck_encoding, ranking_encoding):
        loss = 0.0
        loss += self.missing_card_recovery_loss(deck_with_card_removed, missing_card)
        loss += self.deck_similarity_to_ranking_loss(deck_encoding, ranking_encoding)
        return loss

batch_size = 100
cards = torch.randint(0, NR_OF_CARDS, (batch_size, NR_OF_CARDS_PER_DECK))
rankings = torch.randint(0, NR_OF_BUCKETS, (batch_size, 1))

model = ClashRoyaleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss = 0
for t in range(EPOCHS):
    # Forward pass: Compute predicted y by passing x to the model
    deck_with_card_removed, missing_card, deck_encoding, ranking_encoding = model(cards, rankings)

    # Compute and print loss
    loss = model.compute_loss(deck_with_card_removed, missing_card, deck_encoding, ranking_encoding)

    if t % 10 == 9:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {loss}')