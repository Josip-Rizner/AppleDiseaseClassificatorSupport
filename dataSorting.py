import pandas as pd
import numpy as np
import shutil
import os

data = pd.read_csv("archive/data.csv")
data = pd.DataFrame(data)

data = data.iloc[:,[1,2,3]]
data = data.to_numpy()

labels = data[:, [1]]
unique_labels = np.unique(labels)
sets = np.unique(data[:, [2]])

#Sorting pictures to the classes 
if not os.path.exists("sorted_data"):
    os.mkdir("sorted_data")

for subset in sets:
    path = os.path.join("sorted_data", subset)
    if not os.path.exists(path):
        os.mkdir(path)
    for label in unique_labels:
        path = os.path.join("sorted_data", subset, label)
        if not os.path.exists(path):
            os.mkdir(path)

#Copying images to the correct folders
source = os.path.join(os.getcwd(),"archive", "images")
count = 0

for picture in data:
    file = picture[0]
    file_class = picture[1]
    file_subset = picture[2]
    src = os.path.join(source, file)
    des = os.path.join(os.getcwd(), "sorted_data", file_subset, file_class, file)
    print("source -->" + src)
    print("destination -->" + des)
    shutil.copy(src, des)
    count += 1
    
    
#control
print(count)
