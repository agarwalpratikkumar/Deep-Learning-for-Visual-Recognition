from numpy.core.multiarray import dtype

from load_mnist import *
import hw1_knn  as mlBasics
import numpy as np

#2.a
# Load data - ALL CLASSES
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#2.b
index_store = {}
for i,labels in enumerate(y_train):
    if labels in index_store:
        index_store[labels].append(i)
    else:
        index_store[labels] = [i]

i, j = 0, 100
train_sample = np.empty(1000, dtype='int')
for indexes in index_store:
    train_sample[i:j] = np.random.choice(index_store[indexes], size=100)
    i, j = j, j+100

x_train_sample, y_train_sample = X_train[train_sample], y_train[train_sample]

#Distance Calculation
dists = mlBasics.compute_euclidean_distances(x_train_sample, X_test)

#K=1
y_predicted_k_1 = mlBasics.predict_labels(dists, y_train_sample)
print('{0:0.02f}'.format(np.mean(y_predicted_k_1 == y_test) * 100), "of test examples classified correctly.")

#K=5
y_predicted_k_5 = mlBasics.predict_labels(dists, y_train_sample, k=5)
print('{0:0.02f}'.format(np.mean(y_predicted_k_5 == y_test) * 100), "of test examples classified correctly.")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm_1 = confusion_matrix(y_test, y_predicted_k_1)
print(cm_1)
#plt.imshow(cm_1)
#plt.show()

cm_5 = confusion_matrix(y_test, y_predicted_k_5)
print(cm_5)
#plt.imshow(cm_5)
#plt.show()

# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.imshow(X_test[i])
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("{} ({})".format(y_predicted_k_1[i], y_predicted_k_5[i]))


#2.c
from sklearn.model_selection import KFold

accuracies = []
best_k = []
def Kfold_coss_validate(neighbors = 16, K = 5):
    kfold = KFold(K, True, 1)

    for train_kfold_index, test_kfold_index in kfold.split(x_train_sample):
        train_kfold, test_kfold = x_train_sample[train_kfold_index], x_train_sample[test_kfold_index]
        y_train_kfold, y_test_kfold = y_train_sample[train_kfold_index], y_train_sample[test_kfold_index]

        dists_kfold = mlBasics.compute_euclidean_distances(train_kfold, test_kfold)
        for i in range(1, neighbors):
            y_predicted_kfold = mlBasics.predict_labels(dists_kfold, y_train_kfold, k = i)
            print('{0:0.02f}'.format(np.mean(y_predicted_kfold == y_test_kfold) * 100),
                  "of test examples classified correctly.")
            accuracies.append(np.mean(y_predicted_kfold == y_test_kfold) * 100)
        print("-----------------------------------------")
        best_k.append(np.argmax(accuracies)+1)
        accuracies[:] = []


Kfold_coss_validate()
# for k, acc_all_fold in enumerate(np.array(accuracies)):
#     plt.plot(k, acc_all_fold, label='K '+str(k))
# plt.show()



#2.d
dists_full = mlBasics.compute_euclidean_distances(X_train, X_test)
prediction = mlBasics.predict_labels(dists_full, y_train)
print('{0:0.02f}'.format(np.mean(prediction == y_test) * 100), "of test examples classified correctly.")

print("Best k-neighbors performed on cross validation: ", best_k)


