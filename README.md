# Stock Market Prediction
# The algorithm was implemented without using any Machine Learning libraries.

This project makes use of artificial neural network (ANN) to predict the next day's opening value of Nifty-100 index. The ANN is trained using BackPropagation algorithm and the accuracy of the model is improved by using k-fold Cross Validation Algorithm. The inputs to the neural network are the previous day's closing values of Commodities(Gold,Silver,Copper,Mineral Oil,Natural Gas), Nifty-100 index converted using Simple Moving Average(SMA) and the FOREX rates(USD-INR).

Clone the repository:

```
git clone https://github.com/ak1997/Stock-Market-Prediction.git
```



Execute the script given below to build the neural network,train it and then test it's accuracy.
```
ebpt1(2016 data).py
```

```
training_data = '.../nifty 100 training data.csv'
testing_data = '.../nifty 100 testing data.csv'
```

## Output

This algorithm achieved a maximum test accuracy of 77.8%.
The 'Analysis output.png' file shows the line graph for predicted and actual opening value of the Nifty-100 index.


## Built With

* [Python](https://www.python.org/doc/) - The scripting language used

## Authors

* **Ashutosh Kale** - (https://github.com/ak1997)
* **Omkaar Khanvilkar** - (https://github.com/omkaar23)
* **Hardik Jivani** - (https://github.com/hardikaj96)
* **Ishan Madan** - (https://github.com/ishanmadan1996)
* **Prathamesh Kumkar** - (https://github.com/iPrathamKumkar)
