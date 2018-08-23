# Stock Market Prediction
# The algorithm was implemented without using any Machine Learning libraries. 

This project makes use of artificial neural network (ANN) to predict the next day's opening value of Nifty-100 index.
The ANN is trained using BackPropagation algorithm and the accuracy of the model is improved by using k-fold Cross Validation Algorithm.
The inputs to the neural network are the previous day's closing values of Commodities(Gold,Silver,Copper,Mineral Oil,Natural Gas), Nifty-100 index converted using Simple Moving Average(SMA) and the FOREX rates(USD-INR).
This algorithm achieved a maximum test accuracy of 77.8%.
The 'Analysis output.png' file shows the line graph for predicted and actual opening value of the Nifty-100 index.
