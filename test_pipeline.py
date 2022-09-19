#!/usr/bin/python3

import networkx as nx

from matplotlib import pyplot as plt

from dasf.pipeline.pipeline import Pipeline2


def loader():
    return X

def segarr(X):
    return X + 1

def slicearr(X):
    return X + 2

def envelope(X):
    return X + 10

def sweetness(X):
    return X + 5

def polarity(X):
    return X + 1

def concat_df(**kwargs):
    x = 0
    for k, v in kwargs.items():
        x += v
    return x

def normalize(X):
    return float(X / 10)

class KMeans:
    def fit(self, X):
        return X * -1

kmeans = KMeans()

pipeline = Pipeline2('Test')

pipeline.add(segarr, X=loader) \
        .add(slicearr, X=segarr) \
        .add(envelope, X=slicearr) \
        .add(sweetness, X=slicearr) \
        .add(polarity, X=slicearr) \
        .add(concat_df, envelope=envelope) \
        .add(concat_df, sweetness=sweetness, polarity=polarity) \
        .add(normalize, X=concat_df) \
        .add(kmeans.fit, X=normalize)

pipeline.run()

nx.draw_networkx(pipeline._dag, arrows=True)
plt.savefig("dag.png", format="PNG")
plt.clf()
