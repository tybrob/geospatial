import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
import sys



def spatial_query(df:pd.DataFrame, reference_points:pd.DataFrame, c:float, q:dict(), r:float):
    """
    Given a spatial query with 
        *q.x 
        *q.y 
        *r as distance threshold 
    for each cluster Ci (Ki , ri) that intersects with the circle centered at (q.x, q.y) and radius r
    retriece all the data objects within the interval 
    Ii.low  = i · c + min{dist(Ki , q) - r , 0}
    Ii.high = i · c + max{dist(Ki , q) + r , ri}
    """
    _, ax = plt.subplots()

    query_circle = [q['x'],q['y'],r]
    target_df = pd.DataFrame()

    for idx,row in reference_points.iterrows():
        plt.scatter(row['x'], row['y'], color='black', marker='x')
        ax.add_patch(plt.Circle((row['x'], row['y']), row['r'], color='black',fill=False))
        if check_intersection(query_circle,[row['x'], row['y'], row['r']]):
            print(f"query intersect with cluster C{idx}")
            low,high = calculate_iDistance_spatial_values(idx, c, (q['x'],q['y']), r, row['r'], (row['x'], row['y']))
            spatial_points = get_points_based_on_spatial_iDistance(df.loc[df['a'] == idx], low, high)
            target_df = pd.concat([target_df,spatial_points])
        
    plt.scatter(q['x'],q['y'], color='red', marker='o')
    ax.add_patch(plt.Circle((q['x'],q['y']), r, color='red',fill=False))

    plt.show()

    return target_df


def check_intersection(circle1:list, circle2:list) -> bool:
    """
    check if two circles intersect
    """
    distance = ((circle1[0] - circle2[0]) ** 2 + (circle1[1] - circle2[1]) ** 2) ** 0.5
    radii_sum = circle1[2] + circle2[2]
    if distance <= radii_sum:
        return True
    return False


def calculate_iDistance_spatial_values(i, c, q, r, ri, Ki):
    """
    return the upper and lower bound for iDistance spatial Values
    """
    low = i * c + min(distance(Ki, q) - r, 0)
    high = i * c + max(distance(Ki, q) + r, ri)
    return low, high


def distance(p1, p2):
    """
    calculate the distance of two points
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def get_points_based_on_spatial_iDistance(df, low, high):
    """
    filter spatial_iDistance based on low and high values
    """
    return df.loc[(df['spatial_iDistance'] >= low) & (df['spatial_iDistance'] <= high)]


def textual_query(df:pd.DataFrame, word_partitions:pd.DataFrame, query_keywords:set(), t, c):
    """
    Given a set of keywords Q
    for each partition that has at least one common term with the Q
    we obtain an interval that needs to be searched
    I.low  = i · c' + minscore_i
    I.high = i · c' + maxscore_i
    """
    common_partitions = find_partitions_with_common_terms(word_partitions, query_keywords)
    print(common_partitions)
    v = calculate_vocabulary(word_partitions)
    target_df = df.iloc[:0,:].copy()
    for partition in common_partitions:
        vi = word_partitions.iloc[partition].get("keywords").split("| ")
        minscore = calculate_minscore(query_keywords,vi,v,t)
        maxscore = calculate_maxscore(query_keywords,vi,t)
        (low,high) = calculate_iDistance_textual_values(partition,c, minscore, maxscore)
        df_c = get_points_based_on_textual_iDistance(df,low,high)
        target_df = pd.concat([target_df,get_points_based_on_textual_iDistance(df,low,high)])

    return target_df
        


def find_partitions_with_common_terms(word_partitions:pd.DataFrame, query_keywords:set()):
    """
    find partitions that have at least one common term with Q
    """
    partitions = set()
    for idx, row in word_partitions.iterrows():
        l = row.get("keywords").split("| ")
        for q in query_keywords:
            if (q in l):
                partitions.add(idx)
                break

    return partitions


def calculate_vocabulary(word_partitions:pd.DataFrame):
    """
    return a set with all the keywords
    """
    v = set()
    for _, row in word_partitions.iterrows():
        l = row.get("keywords").split("| ")
        for keyword in l:
            v.add(keyword)
    return v

def calculate_minscore(q:set(), vi:set(), v:set(), t):
    """
        *q: query set
        *vi: partition set
        *v: vocabulary
        *t: textual similarity
    calulate min score
    """
    return t - len(q.intersection(v.difference(vi))) / len(q)


def calculate_maxscore(q:set(), vi:set(), t):
    """
        *q: query sey
        *vi: partition set
        *t: textual similarity
    calculate max score
    """
    return len(q.intersection(vi)) / len(q) +1 -t


def calculate_iDistance_textual_values(i, c, minscore, maxscore):
    """
    return the upper and lower bound for iDistance spatial Values
    """
    low = i * c + minscore
    high = i * c + maxscore
    return low, high


def get_points_based_on_textual_iDistance(df, low, high):
    """
    filter textual_iDistance based on low and high values
    """
    return df.loc[(df['textual_iDistance'] >= low) & (df['textual_iDistance'] <= high)]


if __name__ == '__main__':

    #resthotNA hotel
    file_name = 'resthotNA'
    dict_of_c_values = {'resthotNA':1.3565018881567229E7, 'hotel': 3632.663528619724}
    final_format_hotel = pd.read_csv(f"../data/{file_name}_FinalFormat.txt", sep="|")

    print(len(final_format_hotel))
    word_partitions = pd.read_csv(f"../data/{file_name}_WordPartitions.txt")
    reference_points = pd.read_csv(f"../data/{file_name}_ReferencePoints_2000_new_order.txt", sep=" ")

    #print(final_format_hotel['b'].drop_duplicates().sort_values().to_string())
    #sys.exit()

    #query parameters
    query_spatial = {'x':1134.436174245782,
                     'y':20324.7859924045315}
    r = 10000000.23
    query_textual = {"ski school","skiing","hiking","express check-in/check-out","private check-in/check-out","honeymoon suite","non-smoking rooms","family rooms","english","gyros"}
    t = 0.1

    spatial_points = spatial_query(final_format_hotel, reference_points,dict_of_c_values[file_name],query_spatial,r)
    print(len(spatial_points))
    #TODO try with different query parameters
    textual_points = textual_query(spatial_points, word_partitions, query_textual, t, 1.01)
    print(len(textual_points))

    print(textual_points['a'].drop_duplicates().sort_values().to_string())

    #TODO retrieve the intervals and perform the window queries

    #_, ax = plt.subplots()

    #final_format_hotel = pd.read_csv("../data/resthotNA_FinalFormat.txt", sep="|")

    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 0].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 0].iloc[:,5],color='red')
    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 1].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 1].iloc[:,5],color='blue')
    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 2].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 2].iloc[:,5],color='green')
    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 3].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 3].iloc[:,5],color='yellow')
    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 4].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 4].iloc[:,5],color='black')
    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 5].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 5].iloc[:,5],color='grey')
    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 6].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 6].iloc[:,5],color='purple')
    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 7].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 7].iloc[:,5],color='orange')
    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 8].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 8].iloc[:,5],color='brown')
    plt.scatter(final_format_hotel.loc[final_format_hotel['a'] == 9].iloc[:,4],final_format_hotel.loc[final_format_hotel['a'] == 9].iloc[:,5],color='cyan')

    plt.show()
    
    #a = final_format_hotel.iloc[:,2:4].to_numpy()
    ##find centroids and radius
    #for cluster in final_format_hotel.iloc[:,0].unique():
    #    print(f'cluster={cluster}')
    #    points = final_format_hotel.loc[final_format_hotel['a'] == cluster].iloc[:,2:4].to_numpy()
    #    centroid = np.mean(points, axis=0)
    #    print(f'centroid:{centroid}')
    #    #plt.scatter(*centroid,color='black', marker='x')
    #    distances = np.sqrt(np.sum((points-centroid)**2,axis=1))
    #    max_distance = np.max(distances)
    #    #cir = plt.Circle(centroid, max_distance, color='black',fill=False)
    #    #ax.add_patch(cir)
    #    print(max_distance)
    #plt.show()

"""
q.x q.y Q={set of keywords}
distance threshold r
textual similarity t

for each cluster Ci (Ki , ri) that intersects with the circle centered at (q.x, q.y) and radius r
retriece all the data objects within the interval 
Ii.low  = i · c + min{dist(Ki , q) - r , 0}
Ii.high = i · c + max{dist(Ki , q) + r , ri}
"""