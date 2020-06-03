#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
import warnings
import os
import time
#import nxviz as nv
from scipy import stats
from datetime import datetime
from sklearn import metrics

warnings.filterwarnings('ignore')  # suppress warning messages
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


# Import artist_dictionary.data, tags_dict.data, and yahoo ratings.csv
# from working directory.

artist_dictionary = pickle.load(open('artist_dictionary.data','rb'))
tags_dictionary = pickle.load(open('tags_dict.data','rb'))
filtered_tags_dictionary = pickle.load(open('filtered_tags_dict.data','rb'))
ratings = pd.read_csv('artist-yahoo-ratings.csv')
ratings['artist'] = ratings['artist_name'].str.lower()

spotify_rankings = pd.read_csv('10k_spotify_rankings.csv')

spotify_rankings['Artist'] = spotify_rankings['Artist'].str.lower()

# lower-case all artists - also running at end of cell to make manually adding artists easier

artist_dictionary = {str(key).lower():artist_dictionary[key] for key in artist_dictionary.keys()}

for key in artist_dictionary.keys():
    artist_dictionary[key]=[i.lower() for i in artist_dictionary[key]]

tags_dictionary = {str(key).lower():tags_dictionary[key] for key in tags_dictionary.keys()}

for key in tags_dictionary.keys():
    tags_dictionary[key]=[str(i).lower() for i in tags_dictionary[key]]


filtered_tags_dictionary = {str(key).lower():filtered_tags_dictionary[key] for key in filtered_tags_dictionary.keys()}

for key in filtered_tags_dictionary.keys():
    filtered_tags_dictionary[key]=[str(i).lower() for i in filtered_tags_dictionary[key]]

#remove duplicate values from each key
artist_dictionary = {key:list(set(artist_dictionary[key])) for key in artist_dictionary.keys()}
tags_dictionary = {key:list(set(tags_dictionary[key])) for key in tags_dictionary.keys()}


# In[ ]:


def get_edges(dictionary,label, pop_weight = True, weight = 0):
    """Create a list of edges from a dictionary.

    Create a list of labeled edges for a network graph by comparing the values (artists)
    of each key (band) in a dictionary.  If two values share a key, return a list of
    (key,key,{link:shared value}) tuples.  This function is called by the build_net function to
    generate the edge list for graph construction.

    Parameters
    ----------
    dictionary : dict

    label : str
            Label to be used as key in edge metadata, used to describe node relationship.

    Returns
    -------
    edgelist : list of str and dict

    """
    edgelist_raw = []
    for  key, value in dictionary.items():
        for items in value:
            for key1, value1 in dictionary.items():
                for items1 in value1:
                    if len(items) > 1 and len(items1) > 1: # Prevent single letter artists
                                                         # ('G' or 'K') from linking with
                                                         # every artist with a g or k in their name.                        
                        
                        if items in items1:
                            if pop_weight == True:
                                # adding to the edge attributes the average of the Yahoo! rating count between two artists
                                
                                #solve for df having multiple entires for artist ratings
                                if key in ratings.index:
                                    rating1 = ratings.loc[key, 'yahoo_rating_count'].astype(int)
                                    
                                    # TEMP - there may be multiple artists in the ratings file.  if this is the case,
                                    # select the first value in the series.  Remove when ratings file is consolidated. 
                                    if type(rating1) == pd.core.series.Series:
                                        rating1 = list(ratings.loc[key, 'yahoo_rating_count'])[0]
                                else:
                                    rating1 = 1
                                
                                if key1 in ratings.index:
                                    rating2 = ratings.loc[key1, 'yahoo_rating_count'].astype(int)
                                    
                                    if type(rating2) == pd.core.series.Series:
                                        rating2 = list(ratings.loc[key1, 'yahoo_rating_count'])[0]
                                else:
                                    rating2 = 1
                                
                                avg_rating = (rating1 + rating2) / 2 # average rating between two connected artists
                                                                     # ex. The Beatles (1000) <--> The Who (500)
                                                                     # = 750 edge weight.
                                 
                                norm_rating = avg_rating / 783871    # naive normalize average rating onto a 0 - 1 scale.
                                                                     # 783871 is the highest rating in the Yahoo! data
                                
                               # # check to that the edge does not already exist in edge_list
                               # if (key,key1,{label:items, 'weight':norm_rating}) not in edgelist_raw:
                                
                                edgelist_raw.append((key,key1,{label:items, 'weight':norm_rating}))
                            
                            # return edgelist without edge weight passed to function
                            else:
                                edgelist_raw.append((key,key1,{label:items, 'weight':weight}))

    # remove duplicate edges from edgelist_raw
    edgelist = []
    edgelist = [i for i in edgelist_raw if i not in edgelist]
    
    return edgelist



def build_net(seed, goal=None, size=600, graph=True):
    """ Build an artist-based network graph.

    Create an undirected, artist-based network graph with labeled edges. Basic
    graph structure is equivelant to (band1) <-- shared artist --> (band2). By
    default, function will build networks of size 600 (node count / bands.)
    Optionally, the function can also produce a network that starts with a
    band (seed) and builds a network until it contains the a second band (goal.)
    Warning! There is no check for whether two bands reside in the same network.  It is
    possible to specify a seed and goal for which a relationship does not exist.

    Parameters
    ----------
    seed : str
           Seed band around wihch a network graph is built.  Also root of tree search.

    goal : str
            The function will stop growing the network graph once it contains the goal (band).
            Optional, default = None.

    size : int
           Size of the network graph, defined by node count.
           Optional, default = 600 nodes (bands.)

    graph : bool
            True = return graph object, False = return node/edge lists.

    Returns
    -------
    graph : networkX graph object
            if graph=True, return undirected graph

    nodes,edges : list of str
                  if graph=False, return node and edge lists

    """
    artist_list = []
    band_list = []

    artistindex = 0  # Use index markers to keep track of progress and
    bandindex = 0    # avoid iterating over the entire artist and band lists.

    band_list.append(seed)  # Level 1 seed band (root).

    if goal is not None:

        if nx.has_path(seed,goal) == True:  # Check that a path between seed and goal exists.

            while goal not in band_list:
                for band in band_list[bandindex:]:               # Only retrieve artists for 'new' bands
                    for artist in artist_dictionary[band]:       # in band_list.
                        if artist not in artist_list:
                            artist_list.append(artist)           # Add new artists to artist_list.

                bandindex = len(band_list)                       # Mark current bands in band_list as 'worked'
                                                                 # by setting index marker to the first index
                                                                 # following last element in band_list.

                for artist in artist_list[artistindex:]:         # Only retrieve bands for 'new' artists
                    for key, value in artist_dictionary.items(): # in artist_list.
                        if artist in value:
                            if key not in band_list:
                                if goal not in band_list:
                                    band_list.append(key)        # Add new bands to band_list.

                artistindex = len(artist_list)                  # Mark current artists in artist_list as 'worked'
                                                                 # by setting index marker to the first index
        else:                                                    # following the last element in artist_list.

            print('A path between ',seed,' and ',goal,' does not exist.')


    else:
        prev_band_len = 0
        prev_art_len = 0
        go = True                                   # exit var will flip and end loop if no new artists e
                                                    # are added to the band list (see note below)
        while len(band_list) < size and go == True:

            if len(band_list) > prev_band_len or len(artist_list) > prev_art_len:  
                                                        # check that the len of bandlist has increased, if not
                prev_art_len = len(artist_list)         # set go = no.  added to prevent infinite loops caused
                prev_band_len = len(band_list)          # by artists who have only worked solo ex Madonna.

                for band in band_list[bandindex:]:
                    for artist in artist_dictionary[band]:
                        if artist not in artist_list:
                            artist_list.append(artist)

                bandindex = len(band_list) -1

                for artist in artist_list[artistindex:]:
                    for key, value in artist_dictionary.items():
                        if artist in value:
                            if key not in band_list:
                                if len(band_list) < size:
                                    band_list.append(key)

                artistindex = len(artist_list) -1


            else:
                go = False

    dictionary = {band : artist_dictionary[band] for band in band_list}  # local artist dictionary

    for key, value in dictionary.items():
        dictionary[key] = set(value)   # filter out duplicate group members.

    nodes = list(dictionary.keys())

    edges = get_edges(dictionary, 'Artist')

    for i in edges:              # remove single edge loops from edge list
            if i[0] == i[1]:     # ex. ('The Who','The Who',{'artist':'Roger Daltrey'})
                edges.remove(i)

    if graph == False:

        return(nodes, edges)

    else:
        net_graph = nx.MultiGraph()
        net_graph.add_nodes_from(nodes)
        net_graph.add_edges_from(edges)
        return(net_graph)



# In[ ]:


def add_tag_edges(graph, just_edges=False):
    """Add user tag-based edges to an existing artist network graph.

    Append user tag-based edges to an existing artist graph and return a
    multigraph. Creates edges in the form of (band1,band2,{link:tag})
    tuple.  This function will not add new nodes to the graph.

    Parameters
    ----------
    graph : graph object
            Network graph from which tags will be retrieved, and tag-edges will be added

    just_edges : bool
                 If just_edges = True, return new, undirected multigraph object.
                 If just_edges = False, return list of tag-based edges.

    Returns
    -------
    new_graph : networkX multigraph object

    tag_edges : list of str

    """

    new_graph = nx.MultiGraph()

    #new_graph.add_nodes_from(graph.nodes()) # add nodes from original graph

    new_graph.add_edges_from(graph.edges.data()) # add edges w/attribute dictionary from original graph

    bands = [node for node in graph.nodes()]

    dictionary = {band : tags_dictionary[band] for band in bands
                  if band in tags_dictionary.keys()}  # Check if band exists as a key in
                                                      # in tags_dictionary.

    #if len(dictionary.keys()) >= 2:                   # comparing keys requires atleast two keys
    
    tag_edges = get_edges(dictionary,'User-Tag', pop_weight = False, weight = 0.05) #<<<--- hard-coded tag-edge weight.

    for i in tag_edges:          # remove single edge loops from edge list
        if i[0] == i[1]:         # ex. ('The Who','The Who',{'LINK':'classic rock'})
            tag_edges.remove(i)

    if len(tag_edges) > 0:
        if just_edges == True:

            return tag_edges

        else:
            new_graph.add_edges_from(tag_edges)

            return new_graph


def new_centrality(graph):
    """ Compute centrality scores for a network graph.

    Compute a number of different centrality and misc. scores for all nodes in a network graph.

    Parameters
    ----------
    graph : networkX graph object

    Returns
    -------
    core_df : Pandas DataFrame object

    """

    core_df = pd.DataFrame()
    core_df['artist'] = graph.nodes() # Add to the artist column all nodes (artists) in a graph.
    scores_list = []

    try:
        deg_cent = pd.DataFrame.from_dict(nx.degree_centrality(graph), orient = 'index',  columns = ['deg_cent'])
        scores_list.append(deg_cent)
    except:
        pass

    try:
        load_cent = pd.DataFrame.from_dict(nx.load_centrality(graph), orient = 'index',  columns = ['load_cent'])
        scores_list.append(load_cent)
    #between_cent = nx.betweenness_centrality(graph)
    except:
        pass

    try:
        page_rank = pd.DataFrame.from_dict(nx.pagerank_numpy(graph), orient = 'index',  columns = ['page_rank'])
        scores_list.append(page_rank)
    except:
        pass

    try:
        ev_cent = pd.DataFrame.from_dict(nx.eigenvector_centrality_numpy(graph), orient = 'index',  columns = ['ev_cent'])
        scores_list.append(ev_cent)
    except:
        pass

    try:
        cl_cent = pd.DataFrame.from_dict(nx.closeness_centrality(graph), orient = 'index',  columns = ['close_cent'])
        scores_list.append(cl_cent)
    except:
        pass

    try:
        cfcc = pd.DataFrame.from_dict(nx.current_flow_closeness_centrality(graph), orient = 'index',  columns = ['cf_close_cent'])
        scores_list.append(cfcc)
    except:
        pass
    """
    try:
        ic = pd.DataFrame.from_dict(nx.information_centrality(graph), orient = 'index',  columns = ['info_cent'])
        scores_list.append(ic)
    except:
        pass

        #ebc = pd.DataFrame.from_dict(nx.edge_betweenness_centrality(graph), orient = 'index',  columns = ['edge_bet_cent'])

    try:
        cfbc = pd.DataFrame.from_dict(nx.current_flow_betweenness_centrality(graph), orient = 'index',  columns = ['edge_cflow_cent'])
        scores_list.append(cfbc)
    except:
        pass
    #ecfbc = pd.DataFrame.from_dict(nx.edge_current_flow_betweenness_centrality(graph), orient = 'index',  columns = ['cf_between_cent'])

    try:
        acfbc = pd.DataFrame.from_dict(nx.approximate_current_flow_betweenness_centrality(graph), orient = 'index',  columns = ['appx.cfbt_cent'])
        scores_list.append(acfbc)
    except:
        pass
    #elc = pd.DataFrame.from_dict(nx.edge_load_centrality(graph), orient = 'index',  columns = ['edge_load_cent'])
    """
    try:
        hc = pd.DataFrame.from_dict(nx.harmonic_centrality(graph), orient = 'index',  columns = ['harm_cent'])
        scores_list.append(hc)
    except:
        pass
    #d = pd.DataFrame.from_dict(nx.dispersion(graph), orient = 'index',  columns = ['dispersion'])
    """
    try:
        soc = pd.DataFrame.from_dict(nx.second_order_centrality(graph), orient = 'index',  columns = ['sec_ord_cent'])
        scores_list.append(soc)
    except:
        pass
    """
    df = pd.concat(scores_list, axis = 1)

    core_df = core_df.merge(df,
                            left_on = 'artist',
                            right_index = True)

    core_df['mean_cent'] = core_df.apply(lambda row: np.mean(row[1:]),   #  Calculate the mean of the row
                                    axis = 1)
    return core_df


# In[ ]:


def layer_graphs(artists, add_tags = True,  size = 600):
    """Layer n artist networks together.

    Layer together multiple artist-based graphs, add tag-based edges,
    and return a single multigraph.

    Parameters
    ----------
    args : str
           Band(s) around which network graphs will be built and a layered
           together.
    size : int
           Size in # of nodes for the individual artist networks.

    Returns
    -------
    graph : networkX multigraph object

    """
    layered_graph = nx.MultiGraph()

    for band in artists:
        if band in artist_dictionary.keys():

            artist_graph = build_net(band, size = size)  # Build the artist-based network

            layered_graph.add_edges_from(artist_graph.edges())  # Add edges to layered_graph, nodes
                                                                # will be added if they do not already exist.
        else:
            print(band, ' not in artist_dictionary')

    if add_tags == True:
        layered_graph = add_tag_edges(layered_graph)  # Add tag-based edges to layered_graph and
                                                  # connect the artist graph layers.

    return layered_graph


# In[ ]:


"""  This is a catch-all section for manually adding missing artists and bands
to the artist dictionary.  They will eventually be moved intoartist_dictionary.data.
One day..."""

artist_dictionary['bob seger'] = ['bob seger']
artist_dictionary['tom petty'] = ['tom petty']
artist_dictionary['phil collins'] = ['phil collins']
artist_dictionary['madonna'] = ['madonna']
artist_dictionary['the english beat'] = artist_dictionary['the beat']
artist_dictionary['gloria estefan'] = ['gloria estefan']
artist_dictionary['don henley'] = ['don henley']
artist_dictionary['daryl hall & john oates'] = artist_dictionary['hall & oates']
artist_dictionary['boy george'] = ['boy george']
artist_dictionary['a-ha'] = ['magne furuholmen', 'morten harket','paul waaktaar-savoy']
artist_dictionary['huey lewis & the news'] = artist_dictionary['huey lewis and the news']
artist_dictionary['ilse delange'] =  ['ilse delange']
artist_dictionary['chris isaak'] = ['chris isaak']
artist_dictionary['vanessa carlton'] = ['vanessa carlton']
artist_dictionary['alicia keys'] = ['alicia keys']
artist_dictionary['adele'] = ['adele']
artist_dictionary['amy winehouse'] = ['amy winehouse']
artist_dictionary['rihanna'] = ['rihanna']
artist_dictionary['marco borsato']= ['marco borsato']
artist_dictionary['mary j. blige'] = ['mary j. blige']
artist_dictionary['amy macdonald'] = ['amy macdonald']
artist_dictionary['tom jones'] = ['tom jones']
artist_dictionary['george michael'] = ['george michael']
artist_dictionary['james morrison'] = ['james morrison']
artist_dictionary['iggy pop'] = ['iggy pop']
artist_dictionary['joan osborne'] = ['joan osborne']
artist_dictionary['anti-flag'] = ['andy wright', 'justin sane', 'pat thetic','chris head']
artist_dictionary['kayzo'] = ['kayzo']
artist_dictionary['fame on fire'] = ['bryan kuznitz', 'blake saul', 'paul spirou', 'alex roman']
artist_dictionary['rev theory'] = ['rich luzzi', 'julien jorgensen','rikki lixx','matt mccloskey','dave agoglia']
artist_dictionary['the anix'] = ['brandon smith', 'logan smith','chris dinger']
artist_dictionary['ded'] = ['joe cotela', 'kyle koelsch', 'david ludlow', 'matt reinhard']
artist_dictionary['blind channel'] = ['joel hokka', 'niko moilanen', 'joonas porko', 'olli matela', 'tommi lalli']
artist_dictionary['mammoth mammoth'] = ['mikey tucker', 'ben couzens', 'frank trobbiani','pete bell','gareth sweet','simon jaunay','marco gennaro','kris sinister']
artist_dictionary['dayseeker'] = ['rory rodriguez', 'mike karle', 'alex polk', 'gino sgambelluri','andrew sharp']
artist_dictionary['dream state'] = ['charlotte gilpin', 'aled evans', 'rhys wilcox', 'danny rayer', 'sam harrison-little', 'jamie lee']
artist_dictionary['wage war'] = ['briton bond', 'cody quistad',  'seth blake', 'jordan pierce', 'david rau', 'chris gaylord', 'stephen kluesener']
artist_dictionary['soulja boy'] = ['']
artist_dictionary['fergie'] = ['fergie']
artist_dictionary['pitbull'] = ['pitbull']
artist_dictionary['ne-yo'] = ['ne-yo']
artist_dictionary['jessie j'] = ['jessie j']
artist_dictionary['nicki minaj'] = ['nicki minaj']
artist_dictionary['jennifer lopez'] = ['jennifer lopez']
artist_dictionary['chris brown'] = ['chris brown']
artist_dictionary['justin bieber'] = ['justin bieber']
artist_dictionary['taio cruz'] = ['taio cruz']
artist_dictionary['jason derulo'] = ['jason derulo']
artist_dictionary['flo rida'] = ['flo rida']
artist_dictionary['iyaz'] = ['iyaz']
artist_dictionary['black eyed peas'] = artist_dictionary['the black eyed peas']
artist_dictionary['alexandra stan'] = ['alexandra stan']
artist_dictionary['david guetta'] = ['david guetta']
artist_dictionary['fidlar'] = ['zac carper', 'elvis kuehn', 'max kuehn','brandon schwartzel']

artist_dictionary["swingin' utters"] = ['kevin wickersham','greg mcentee', 'darius koski', 'max huber', 'spike slawson','jack dalrymple', 'miles peck']
artist_dictionary['the 4-skins'] = ['steve harmer','gary hodges','tom mccourt']
artist_dictionary['4 skins'] = artist_dictionary['the 4-skins']
artist_dictionary['jfa']=['Mike Tracy','Brian Brannon','Don Pendleton','Corey Stretz','Todd Barnes',                          'Alan Bishop','Scott Chazan','Michael Cornelius','Bob Cox','Brian Damage',                          'Joel DuBois','Trace Element','Matt Etheridge','Jim Moore','Don Pendelton',                          'Al Penzone','Jaime Reidling','Mike Sversvold','Bruce Taylor','Mike Tracy']
artist_dictionary['nihilistics'] = ['Dave Phinster','Mike King','Ron Rancid','Ajax Lepinski']
artist_dictionary['the neighborhoods'] = ['Dave Minehan','John Hartcorn', 'Michael Quaglia']
artist_dictionary['nobodys'] = ['Geoff Palmer', 'J. J. Nobody', 'Justin Disease', 'Randy The Kid']
artist_dictionary['china white'] = ['Marc Martin','James Rodriguez','Joey Ruffino','Frank Ruffino',                                    'Scott Sisunik','Richard Katchadoorian','Vince Mesa','Corey Stretz',                                    'Steven Barrios','Jeff Porter','James Lugo','Sandy Hancock']
artist_dictionary['the fairlanes'] = ['Andy Baldwin', 'Jason Zumbrunnen', 'Robbie Kalinowski', 'Scott Weigel']
artist_dictionary['pink lincolns'] = ['Chris Barrows','Dorsey Martin','Kevin Coss','Jeff Fox']
artist_dictionary['zeke'] = ['Blind Marky Felchtone','Kyle Whitefoot','Kurt Colfelt','Dayne Porras',                             'Jeff Hiatt','Chris Johnsen','Buzzy','Kurt Colfelt','Jeff Matz',                             'Mark Pierce','Abe Zanuel Riggs III','Dizzy Lee Roth','Donny Paycheck']
artist_dictionary['x-ray spex'] = ['B.P. Hurding','Poly Styrene','Steve Thompson','Lora Logic']
artist_dictionary['fidlar'] = ['Zac Carper','Elvis Kuehn','Max Kuehn','Brandon Schwartzel']
artist_dictionary['johnny thunders & the heartbreakers'] = artist_dictionary['the heartbreakers']
artist_dictionary['mike v & the rats'] = ['Jason Hampton', 'Mike Vallely', 'P.T. Pugh', 'Reid Black']
artist_dictionary['g-eazy'] = ['g-eazy']
artist_dictionary['post malone'] = ['post malone']
artist_dictionary['lil skies'] = ['lil skies']
artist_dictionary['lil mosey'] = ['lil mosey']
artist_dictionary['youngboy never broke again'] = ['youngboy never broke again']
artist_dictionary['a boogie wit da hoodie'] =['a boogie wit da hoodie']
artist_dictionary['lil baby'] = ['ll baby']
artist_dictionary['pnb rock'] = ['pnb rock']
artist_dictionary['tee grizzley'] = ['tee grizzley']
artist_dictionary['meek mill'] = ['meek mill']
artist_dictionary['lil zay osama'] = ['lil zay osama']
artist_dictionary['montana of 300'] = ['montana of 300']
artist_dictionary['playboi'] =['playboi']
artist_dictionary['strap da fool'] = ['strap da fool']
artist_dictionary['lil durk'] = ['lil durk']
artist_dictionary['juice wrld'] = ['juice wrld']
artist_dictionary['quando rondo'] = ['quando rondo']
artist_dictionary['yk osiris'] = ['yk osiris']
artist_dictionary['pnb meen'] = ['pnb meen']
artist_dictionary['lil uzi vert'] = ['lil uzi vert']
artist_dictionary['knuckle puck'] =['Joe Taylor','Kevin Maida','John Siorek','Nick Casasanto','Ryan Rumchaks']
artist_dictionary['the story so far'] = ['Parker Cannon','Kelen Capener','Kevin Geyer','Will Levy','Ryan Torf']


artist_dictionary = {str(key).lower():artist_dictionary[key] for key in artist_dictionary.keys()}
for key in artist_dictionary.keys():
    artist_dictionary[key]=[i.lower() for i in artist_dictionary[key]]
