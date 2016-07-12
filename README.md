# ProjectDeepLearning
This Project is made of C++
to learn how deep learning mechanism works.
Currently, it is based on MLP, using SVM loss function,
and optimizes via gradient descent.
The program only can read MNIST dataset,
but you can add new functions at "dataread.cpp"
to load more libraries.

==How to Use==

1. Compiled program must be in the same directory with dataset(e.g. MNIST)
2. Choose which set to train.
3. Choose whether you want to use validate set or not.
4. If so, choose which hyperparameter to validate.
5. Choose whether you want to start new, or load previous weights.
6. During training, you can input some commends.
7. Input Q for quit.
8. Your trained weight parameters are saved at "TrainedParam" file.
