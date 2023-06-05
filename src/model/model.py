import torch
import torch.nn.functional as F

from src.constants import *

class ClashRoyaleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nr_of_cards = NR_OF_CARDS
        self.nr_of_ranking_buckes = NR_OF_BUCKETS
        self.nr_of_cards_per_deck = NR_OF_CARDS_PER_DECK
        self.nr_of_neurons_per_dense = DENSE_NEURONS
        self.nr_of_card_aggregate_features = CARD_ENCODING_SIZE
        self.encoding_size = ENCODER_DIM
        self.input_size = self.nr_of_cards + self.nr_of_ranking_buckes

        self.encoding_layer = torch.nn.Linear(
            in_features=self.input_size, 
            out_features=self.encoding_size,
            bias=False,
            dtype=torch.float32
        )
        self.dense_layer_1 = torch.nn.Linear(
            in_features=self.nr_of_card_aggregate_features,
            out_features=self.nr_of_neurons_per_dense,
            bias=True,
            dtype=torch.float32
        )
        self.dense_layer_2 = torch.nn.Linear(
            in_features=self.nr_of_neurons_per_dense,
            out_features=self.nr_of_neurons_per_dense,
            bias=True,
            dtype=torch.float32
        )
        self.classifier_layer = torch.nn.Linear(
            in_features=self.nr_of_neurons_per_dense,
            out_features=self.input_size,
            bias=True,
            dtype=torch.float32
        )

        self.loss = torch.nn.MSELoss()
        self.loss_target = torch.zeros((1, self.encoding_size))

    def forward(self, deck_encoding: torch.Tensor, ranking_encoding: torch.Tensor, cards_remaining_aggregated_data: torch.Tensor, card_removed_encoding: torch.Tensor, bucket_encoding: torch.Tensor):
        """
        Inputs:
            cards - Tensor of shape (b, nr_of_cards_per_deck), where last dimension is a list of int indices of cards in the deck
            ranking - Tensor of shape (b, 1), where last dimension is the ranking number, int in range [0,NR_OF_BUCKETS)
        """

        deck_output_encoding = self.encoding_layer(deck_encoding)
        ranking_encoding = self.encoding_layer(ranking_encoding)
        deck_ranking_diff = deck_output_encoding - ranking_encoding

        aggregated_data = self.dense_layer_1(cards_remaining_aggregated_data)
        aggregated_data = torch.relu(aggregated_data)
        aggregated_data = self.dense_layer_2(aggregated_data)
        aggregated_data = torch.relu(aggregated_data)
        aggregated_data = self.classifier_layer(aggregated_data)
        aggregated_data = torch.softmax(aggregated_data, -1)

        aggregated_data_output_encoding = self.encoding_layer(aggregated_data)
        card_removed_output_encoding = self.encoding_layer(card_removed_encoding)
        cards_diff = aggregated_data_output_encoding - card_removed_output_encoding

        bucket_output_encoding = self.encoding_layer(bucket_encoding)
        bucket_output_encoding = bucket_output_encoding - deck_output_encoding
        bucket_output_encoding = bucket_output_encoding * bucket_output_encoding
        bucket_output_encoding = torch.sum(bucket_output_encoding, -1)
        distances = torch.sqrt(bucket_output_encoding)
        bucket = torch.argmax(distances)

        return deck_ranking_diff, cards_diff, bucket

    def compute_loss(self, deck_ranking_diff, cards_diff, bucket, target_bucket):
        return 1e3*(self.loss(deck_ranking_diff, self.loss_target) + self.loss(cards_diff, self.loss_target)) + torch.abs(bucket - target_bucket)/self.nr_of_ranking_buckes

    def get_encoding(self, card_idx):
        card_encoding: torch.Tensor = self.encoding_layer.weight[:, card_idx]
        return card_encoding.detach().numpy()

    def predict(self, input):
        return self.encoding_layer(input).detach().numpy()
    
def get_model_with_optimizer():
    model = ClashRoyaleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, optimizer