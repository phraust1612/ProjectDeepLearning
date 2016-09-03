# ProjectDeepLearning

This Project is composed of C++.
And the purpose is to study how deep learning mechanism works.
Currently, it is based on CNN,
using SVM for loss function,
and optimizes by momentum update.
It doesn't support cuda yet,
but it's able to run in multi-thread via CPU.

===========================How to Use============================

0. Call "PDL.exe (saved file name)" in console to load previous file
   or call "PDL.exe" in console to start newly.
   You can just execute saved files directly in Windows GUI.
1. Choose which dataset to use.
2. Choose whether to use validation set or not.
3. Input your file's name if you're starting for the first time.
4. Set your hyperparameters.
5. During training, you can input some commends (press h for help)
6. Press 'q' to end the program.

=========================Hyperparameters=========================

if B>0 : [(CONV -> ReLU) * A -> POOL?] * B -> (FC -> ReLU) * C -> FC
if B=0 : (CONV -> ReLU) * A -> (FC -> ReLU) * C -> FC

Every Conv and Pooling layer requires
Filter(F), Stride(S), Zero-padding(P), and depth of the layer.
And every FC layer requires its Dimension(D) except final score layer.

You must decide values of above hyperparameters.
Below is a list of additional hyperparameters you can use.

Delta is a hyperparameter used at SVM.
Lambda is a hyperparameter used at L2 regularization.
H is the learning rate.
Momentum update constance is a hyperparameter used at optimization.

===========================Patch Notes===========================

ver 2.2
   - Now it supports CNN, but Pooling layer doesn't work yet
   - Gradients of conv layer weights are seem to be wrong
     Accuracy ascends at the first epoch but it fails after.
   - Still has a mathematical error in gradient checking
   - WeightSave should be fixed properly on the basis of Conv weights
   - Modifying command should be fixed.

ver 2.2.1
   - CheckAccuracy function now supports multi-threading
     but it occurs an error when it's RNN
   - Testindex function is newly added
     I checked there's no matter with indexing functions