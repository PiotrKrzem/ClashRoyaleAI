import torch
import torch.nn.functional as F

from src.model.model import get_model_with_optimizer
from src.model.utils import get_training_testing_data, get_ranking_vectors, compute_distances
from src.constants import *

def tfn(mtx):
    return torch.from_numpy(mtx)

def test_model(model, test_decks, ):
    ranking_encodings = get_ranking_vectors()
    ranking_embeddings = model.predict(tfn(ranking_encodings))
    total_predictions = 0
    correct_predictions = 0
    for deck in test_decks:
        deck_encoding = deck.one_hot_deck()
        deck_embedding = model.predict(tfn(deck_encoding))
        predicted_bucket = compute_distances(deck_embedding, ranking_embeddings)
        # print(deck_embedding, predicted_bucket, deck.rank//(MAX_RANKING//NR_OF_BUCKETS))
        target_bucket = deck.rank//(MAX_RANKING//NR_OF_BUCKETS)
        if predicted_bucket in [target_bucket, target_bucket - 1, target_bucket + 1]:
            correct_predictions += 1
        total_predictions += 1
    print(f"Test accuracy: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions})")
    
def train_model(model, optimizer, training_decks, test_decks, weights):
    loss = 0
    ranking_encodings = get_ranking_vectors()
    test_model(model, training_decks)
    for t in range(EPOCHS):
        for deck in training_decks:
            deck_encoding = deck.one_hot_deck()
            ranking_encoding = deck.one_hot_ranking()
            cards_remaining_aggregated_data, card_removed_encoding = deck.split(weights)
            # Forward pass
            deck_ranking_diff, cards_diff, bucket = model(tfn(deck_encoding), 
                                                tfn(ranking_encoding), 
                                                tfn(cards_remaining_aggregated_data), 
                                                tfn(card_removed_encoding),
                                                tfn(ranking_encodings))
            # Compute and print loss
            target_bucket = deck.rank//(MAX_RANKING//NR_OF_BUCKETS)
            loss = model.compute_loss(deck_ranking_diff, cards_diff, bucket, target_bucket)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % 10 == 9:
            print(f"Epoch: {t}, Loss: {loss.item()}")
            test_model(model, training_decks)

    print(f'Result: {loss}')
    test_model(model, test_decks)

model, optimizer = get_model_with_optimizer()
training_decks, test_decks, cards, weights = get_training_testing_data()
train_model(model, optimizer, training_decks, test_decks, weights)
