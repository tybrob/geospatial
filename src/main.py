import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


final_format_hotel = pd.read_csv("../data/hotel_FinalFormat.txt", sep="|")
final_format_resthotNA = pd.read_csv("../data/resthotNA_FinalFormat.txt", sep="|")

print(final_format_hotel)
#print(final_format_resthotNA)

#plt.scatter(final_format_hotel.iloc[:,2],final_format_hotel.iloc[:,3],color='red')

#plt.scatter(final_format_resthotNA.iloc[:,2],final_format_resthotNA.iloc[:,3],color='blue')

print(final_format_hotel.iloc[:,2:4])

a = final_format_hotel.iloc[:,2:4].to_numpy()

print(a)

#find centroids and radius
print(final_format_hotel.iloc[:,0].unique())

for cluster in final_format_hotel.iloc[:,0].unique():
    print(f'cluster={cluster}'," ")
    points = final_format_hotel.loc[final_format_hotel['a'] == cluster].iloc[:,2:4].to_numpy()
    centroid = np.mean(points, axis=0)
    print(f'centroid:{centroid}')
    distances = np.sqrt(np.sum((points-centroid)**2,axis=1))
    max_distance = np.max(distances)
    print(max_distance)
#plt.show()