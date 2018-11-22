__Linear Classifier with Logistic Function__

In this classification model we have following linear mapping: h(X,W,b) = sigma(WX + b) <br>
Sigmoid Function: h(a) = sigma(a) = 1 / 1 + exp(-a) <br>

![loss_derived](https://user-images.githubusercontent.com/41232373/48922489-3f13a000-eea7-11e8-95f4-9f17983ccccc.jpg)

This is the partial derivatives of the loss function with respect to W and b.
Now we can calculate our gradient with respect to W and the gradient with respect to b.
