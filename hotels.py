import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

"""Read file in as f."""
#df = pd.read_csv(f, header = 0)
"""In this instance, just using the two sample lines provided."""
df = pd.DataFrame(np.array([[1,-0.118092,51.509865], [2,-73.935242,40.730610]]), columns = ['hotel_id', 'longitude', 'latitude'])

"""Create column for the coordinates based on longitude and latitude."""
df['coord'] = df[['latitude', 'longitude']].values.tolist()

"""Create an array of those coordinates, to use with BallTree."""
Y = np.array(df['coord'].values.tolist())

"""Initialise tree."""
tree = BallTree(Y, leaf_size=15)

"""Compute the distances and indices of the nearest k points within the tree."""
dist, ind = tree.query(Y, k=2)

"""Print the indices (or alternatively, relate them to the hotel names and then join that to the database for a list of the nearest K hotels per hotel."""
print(ind)

"""The closest hotel according to this method will always be the same hotel (index 0), but this can easily be filtered out before adding these columns to the df."""