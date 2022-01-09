# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 17:30:20 2022

@author: sonig
"""

import pandas as pd
import numpy as np
import streamlit as st 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

df_ = pd.read_csv("Bollywood songs.csv", encoding='cp1252')
df_ = df_.iloc[: , 1:]
df_.rename(columns={'Song-Name' : 'songs',
                        'Singer/Artists':'Singer',
                        'Album/Movie':'Album',
                        'User-Rating':'Ratings'}, inplace=True)
st.title('Song Recommendation System')
st.subheader('Please select your choice of song')
new_song = st.selectbox('List of songs',(df_['songs']))
st.write('You selected:', new_song)

df_['Genre'] = df_['Genre'].apply(lambda x:x.replace("Bollywood"," "))
df_['Ratings'] = df_['Ratings'].apply(lambda x:x.replace("/10"," "))
df_['Ratings'] = pd.to_numeric(df_['Ratings'], errors='coerce')
df_.Ratings.fillna(value = df_.Ratings.mean(), inplace=True)
df1 = df_.drop_duplicates()

df_songs_features = df1.pivot_table(index='songs', columns='Genre', values='Ratings').fillna(0)
X = csr_matrix(df_songs_features.values)
num = len(df1['songs'].unique())
song_mapper = dict(zip(np.unique(df1["songs"]), list(range(num))))
song_inv_mapper = dict(zip(list(range(num)), np.unique(df1["songs"])))

def find_similar_songs(new_song, X = X, k = 10, metric='cosine', show_distance=False):
    neighbour_ids = []
    song_ind = song_mapper[new_song]
    song_vec = X[song_ind]
    k+=1    
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    song_vec = song_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(song_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(song_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

Recom = find_similar_songs(new_song)
st.write(Recom)

