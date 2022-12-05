import pandas as pd

ref_points_hotel = pd.read_csv("../data/hotel_ReferencePoints_2000_new_order.txt")
ref_points_resthotNA = pd.read_csv("../data/resthotNA_ReferencePoints_2000_new_order.txt")

print(ref_points_hotel)
print(ref_points_resthotNA)

final_format_hotel = pd.read_csv("../data/hotel_FinalFormat.txt", sep="|")
final_format_resthotNA = pd.read_csv("../data/resthotNA_FinalFormat.txt", sep="|")

print(final_format_hotel)
print(final_format_resthotNA)