Deep Learning
-------------

Geoffrey Hinton "The Godfather of Deep Leaning" done research in 80's.  

Mimic how the human brain "one of the most powerful tool in the world for learning" operates. 

Input Layer(starting point) -> Hidden Layer (Processing Layer) -> Output Layer(Prediction)

Multiple hidden layers -> Deep learning. 

Single hidden layer -> Shallow learning. 


The Input Layer
---------------

Contain inputs for single entity " X1: Name, X2: Age, X3: Salary, X4: Sex ".

The Output Layer
----------------

Could be -> contineous, categorical (multiple outputs) or banary

Weights
-------

This is how a neural network learned by adjusting them. Which signal is poor and which are important. 

The Neuron
----------

Neuron computes the weighted sum of all the input values (∑wixi). Then it applies activation function over the weighted sum. By end it decides the neuron passes/not passes. 

The Activation Function
-----------------------

1. Threshold Function: Kind of Binary (if x >= 0 -> 1 elseif x < 0 -> 0)

2. The Sigmoid Function: Specialy useful in output layers if predicting probabilities. 

3. The Rectifier Function: Most popular for neural networks. From 0 to input value gradually progresses. It remains zero until some point the shoots up. As in the case of builing, when it reaches up 100 years its demand increases. 

4. Hyperbolic Tangent Function: Simillar to Sigmoid function but it will go below zero (-1 -> 0 -> 1).

How do NNs Work
---------------

Why each input is not connected to every hidden layer neuron? The reason could be each input contains different values i.e: zero/non-zero values. Also may be the first neuron in hidden layer is looking for inputs like area and distance from city and not interested into number of bedrooms or the gae of property. 

How do NNs Learn
----------------

Cost Function: What is the error you have in your prediction and out goal is to minimize the cost function. 

After calculating the cost the details goes back into the neuron and finally weights get updated. The only control we have in NN are the weights.

Gradient Descent
----------------

Descending into the minimum of the cost function. 

Stochastic Gradient Descent
---------------------------

In normal we can stuck in local minima and surely will not get the optimal one which is the best. This way we will get part of NN. 

In batch mode we take the whole batch apply to NN and run that. In stochastic we take row by row/mini batch from sample run NN then adjust weight. 

Backpropogation
---------------
During process of backpropogation, the way the algorithm is structured we can adjust all the weights at the same time. 

 





















