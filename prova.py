import cv2
import pandas as pd
from os.path import join

DATAPATH = r"C:\Users\anton\Desktop\PROGETTI\EyeGazeRedirection\MPIIGaze"

df = pd.read_csv(join(DATAPATH,"annotation_subset","gaze.csv"))

print(df.head())


for j in range(len(df)):
    line = df.iloc[j]
    eye_region = line.eye_region_details
    eye_details = line.eye_details
    otherpat = line.path
    image_path = line.image_path.rsplit("\\",2)[-1]
    load = cv2.imread(join(DATAPATH,"Images",image_path))
    print(line)