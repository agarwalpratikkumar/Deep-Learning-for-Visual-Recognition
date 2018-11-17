__K-Nearest Neighbors__ (A classification technique)<br>

![knn](https://user-images.githubusercontent.com/41232373/48345734-24585480-e679-11e8-86d0-bc907115d163.png)


__Working:__
From the diagram above, lets Consider the circles and the triangles are the training samples. Circles belong to class Red and Triangles belongs to class Yellow. And let's consider the blue star is the test data.

__Goal:__ Our aim is to predict the class of the test data. <br>

__Steps:__<br>
1. Calculate the euclidean distance between the test data and all the training data. And arrange them in ascending order.
2. In the name K-nearest neighbors, K takes an integer value. If the value of K = 3 then we have to consider 3 nearest neighbours of our test data. We can get these nearest neighbors from the distances which we have calculated. In our diagram we have choosen K = 5 (shown by that circle).
3. Look at the classes of all the nearest neighbors. Test data will belong to that class which will be in majority. So in our case, there are 3 yellow class and 2 Red class hence we can predict that the test data (i.e the blue star) belongs to Yellow class.

__Note:__ We don't know beforehand which K value will perform best. We have to try with different values and check the accuracies.
