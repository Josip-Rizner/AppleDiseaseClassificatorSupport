import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
data = pd.DataFrame(data)

data = data.iloc[:,[1,2,3]]
data = data.to_numpy()

#labels = data[:, [1]]
#unique_labels = np.unique(labels)
#sets = np.unique(data[:, [2]])

unique_labels = ["complex", "rust frog_eye_leaf_spot", "frog_eye_leaf_spot", "frog_eye_leaf_spot complex", "healthy", "powdery_mildew", "powdery_mildew complex", "rust", "rust complex", "scab", "scab frog_eye_leaf_spot", "scab frog_eye_leaf_spot complex"]



pictures_per_class_train_dataset = [0] * 12
pictures_per_class_validation_dataset = [0] * 12
pictures_per_class_test_dataset = [0] * 12
total = [0] * 12

for picture in data:
    picture_class = picture[1]
    picture_subset = picture[2]
    
    for index, label in enumerate(unique_labels):
        if label == picture_class:
            total[index] += 1
    
    if picture_subset == "test":
        for index, label in enumerate(unique_labels):
            if label == picture_class:
                pictures_per_class_test_dataset[index] += 1
                
    elif picture_subset == "train":
        for index, label in enumerate(unique_labels):
            if label == picture_class:
                pictures_per_class_train_dataset[index] += 1
    
    elif picture_subset == "valid":
        for index, label in enumerate(unique_labels):
            if label == picture_class:
                pictures_per_class_validation_dataset[index] += 1
                



#fig, (barchart, pie) = plt.subplots(2)
#barchart.barh(unique_labels, pictures_per_class_train_dataset, height = 0.9)
#pie.pie(pictures_per_class_train_dataset, labels = unique_labels, shadow = True)

plt.figure(1)
plt.barh(unique_labels, total, height = 0.9)
plt.show()


plt.figure(2)
plt.pie(total, labels = unique_labels, shadow = True)
plt.show()

total = np.array(total)
pictures_per_class_train_dataset = np.array(pictures_per_class_train_dataset)
pictures_per_class_validation_dataset = np.array(pictures_per_class_validation_dataset)
pictures_per_class_test_dataset = np.array(pictures_per_class_test_dataset)


#control
print(total.sum())
print(pictures_per_class_train_dataset.sum())
print(pictures_per_class_validation_dataset.sum())
print(pictures_per_class_test_dataset.sum())
