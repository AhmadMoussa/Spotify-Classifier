from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

data = np.array(pd.read_csv("data.csv"))

print(data.shape)

data_train = []
for i in range(1,14):
    temp = data[:511,i]
    temp = np.append(temp, data[1022:1533,i])
    data_train.append(temp)

features_train = pd.DataFrame(data_train)
features_train = features_train.transpose()
print(features_train.shape)


labels_train = data[:511,14]
labels_train = np.append(labels_train, data[1022:1533,14])
labels_train = labels_train.astype('int')

print("Training Labels Shape: ", len(labels_train))

data_test = []
for i in range(1,14):
    temp = data[511:1022,i]
    temp = np.append(temp, data[1533:,i])
    data_test.append(temp)

features_test = pd.DataFrame(data_test)
features_test = features_test.transpose()
print(features_test.shape)

labels_test = data[511:1022,14]
labels_test = np.append(labels_test, data[1533:,14])
labels_test = labels_test.astype('int')
print("Testing Labels Shape: ", len(labels_test))
#1022 is where songs that I don't like start
#divide data set for an equal split for training and testing


clf = DecisionTreeClassifier(min_samples_split = 40)

clf.fit(features_train, labels_train)
print("Trained Model")
#clf.predict(features_test)

print("Accuracy of our model: ",clf.score(features_test,labels_test))
