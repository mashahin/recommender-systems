import tensorflow as tf
from typing import Dict, Tuple
from typing import Dict, Text
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import matplotlib.pyplot as plt


# Ratings data.
ratings_data = tfds.load('movielens/100k-ratings', split="train")
# Features of all the available movies.
features_data = tfds.load('movielens/100k-movies', split="train")

# Select the basic features.
ratings_data = ratings_data.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

# Convert user_ids and movie_title into integers
features_data = features_data.map(lambda x: x["movie_title"])
users = ratings_data.map(lambda x: x["user_id"])

user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(
    mask_token=None)
user_ids_vocabulary.adapt(users.batch(1000))

movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(
    mask_token=None)
movie_titles_vocabulary.adapt(features_data.batch(1000))

# Group by user_id.


def key_func(x): return user_ids_vocabulary(x["user_id"])
def reduce_func(key, dataset): return dataset.batch(100)


train = ratings_data.group_by_window(
    key_func=key_func, reduce_func=reduce_func, window_size=100)

# Here we can check the shape of our data.
print(train)
for x in train.take(1):
    for key, value in x.items():
        print(f"Shape of {key}: {value.shape}")
        print(f"Example values of {key}: {value[:5].numpy()}")
        print()

# Generating batch of labels and features


def _features_and_labels(
        x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    labels = x.pop("user_rating")
    return x, labels


train = train.map(_features_and_labels)

train = train.apply(
    tf.data.experimental.dense_to_ragged_batch(batch_size=32))


# Define the model.
class RankingModel(tf.keras.Model):

    def __init__(self, user_vocab, movie_vocab):
        super().__init__()
        self.user_vocab = user_vocab
        self.movie_vocab = movie_vocab
        self.user_embed = tf.keras.layers.Embedding(
            user_vocab.vocabulary_size(), 64)
        self.movie_embed = tf.keras.layers.Embedding(
            movie_vocab.vocabulary_size(), 64)

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:

        embeddings_user = self.user_embed(self.user_vocab(features["user_id"]))
        embeddings_movie = self.movie_embed(
            self.movie_vocab(features["movie_title"]))

        return tf.reduce_sum(embeddings_user * embeddings_movie, axis=2)


# Compile the model
model = RankingModel(user_ids_vocabulary, movie_titles_vocabulary)
optimizer = tf.keras.optimizers.Adagrad(0.5)
loss = tfr.keras.losses.get(
    loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True)
eval_metrics = [
    tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True),
    tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=True)
]
model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)


# Fit the model
history = model.fit(train, epochs=9)
print("\n", history.history)

# Generate predictions
for movie_titles in features_data.batch(2000):
    break

inputs = {
    "user_id":
        tf.expand_dims(tf.repeat("26", repeats=movie_titles.shape[0]), axis=0),
    "movie_title":
        tf.expand_dims(movie_titles, axis=0)
}

scores = model(inputs)
titles = tfr.utils.sort_by_scores(scores,
                                  [tf.expand_dims(movie_titles, axis=0)])[0]
print(f"Top 10 recommendations for user 26: {titles[0, :10]}")
