# graph_conv_network
Exploring Graph Convolutional Networks for artist-based network graphs and music recommendation.

## What's in the repo:

### Data files
* 10k_spotify_rankings.csv - Spotify stream count for the top 10k artists.
* playlist_graph_artists_scores.csv 
* artist-yahoo-ratings.csv - Yahoo! artist ratings data.
* artist_dictionary.data  - Core artist dictionary.
* filtered_tags_dict.data - User-tag dictionary filtered to the most common genres.
* tags_dict.data - Core user-tag dictionary.

### Notebooks
* GCN.ipynb - Initial exploration of StellarGraph's implementation of GCN.
     - https://github.com/stellargraph/stellargraph


### /music_graph
* All of the core graph-building code written for the original graphing project.  Code has been packaged to make importing the necessary functions and data cleaner, consistent, and more efficient across multiple experimenting and evaluation notebooks.
