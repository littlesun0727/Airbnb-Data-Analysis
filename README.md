# Airbnb-Data-Analysis

Members: Jing Wang (jw5665), Zhirui Liang (zl3364)

Our dataset: reviews_detail(1050548x6), listings_detailed(50041X96), rolling_sales(82145x20)

## [Data Cleaning](https://github.com/littlesun0727/Airbnb-Data-Analysis/blob/main/Data%20cleaning/Visualization.ipynb)
(Dataset from [Airbnb](http://insideairbnb.com/get-the-data.html) & [NYU Rolling Sales Data](https://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page))
- **Handle ‘bad’ data**: we import the re library to transform price-related columns into int or float type. For
example, by re.findall(r'[0-9]+, string), change ‘$ 100000’(str) into 100000(int).
- **Remove ‘missing’ data**: Use dropna(inplace=True) remove all NA rows.

## [Machine Learning](https://github.com/littlesun0727/Airbnb-Data-Analysis/blob/main/Linear%20Regression/Price_estimate_with_review.ipynb)
### Categorical Variables in regression
We try to fit a pricing model for our dataset. Check all decision variables firstly:

- accommodates (float), bathrooms (float), bedrooms(float), review_scores_rating (float), security_deposite
(float), cleaning_fee (float), beds (float), number_of_reviews (float), host_listings_count(float),
neighbourhood (string), room_type (string).

We find variables ‘neighbourhood’ and ‘room type’ are two string columns, which is difficult for our regression
analysis. In order to solve it, we come up with 2 methods:

  - Transfer string into float, assign an integer value to every distinct string.
  - Keep them as categorical variables.
  
With method 1), there are a large number of regression models we can apply, such as Linear regression,
Lasso regression, and Ridge regression. However, this method is not suitable when we dig into the actual
meaning of variables. For example, variable ‘neighbourhood’ means the location information of a house. It is
meaningless to compare the value of this variable after we simply transform it from string type into float
type. With method 2), we can cluster the data by variable ‘neighborhood’, which is more reasonable. In order
to include categorical variables in the regression, we apply a method named categorical regression, which is
a model based on ordinary linear regression.  

### Modify Categorical Regression
As ordinary linear regression always has a large bias, we find a method to minimize the error. In fact, the
linear regression on variables satisfying Normal distribution always performs better than other cases. Therefore, we
transform the original dataset using square and log function to remove the skewness of all variables:

<img width="400" height="400" src="https://github.com/littlesun0727/Airbnb-Data-Analysis/blob/main/Linear%20Regression/QQ%20plot%20of%20original%20dataset.png"> <img width="400" height="400" src="https://github.com/littlesun0727/Airbnb-Data-Analysis/blob/main/Linear%20Regression/QQ%20plot%20of%20transformed%20dataset.png">

From the QQ plot, we can conclude we successfully make all variables approximately satisfy normal distribution.
What’s more, the MSE of the model with transformation decreases from 0.41 to 0.33 compared with the model
without transformation. Thus, our transformation increases the prediction accuracy a lot.

To make our model more reality-oriented, we scale the variables. As all know, coefficients in linear regression
can reflect the importance of corresponding variables if and only if all the variables are on the same scale. For
example, variable ‘review score rating’ belongs to interval (0, 100) while variable ‘accommodates’ in interval
(0, 15). As a result, larger parameter of ‘review score rating’ dosen’t represent it is more important than
‘accommodates’. For the purpose of fixing this problem,, we scale all variables into interval (0, 1). Then the
larger the coefficient of a variable, the more important of this factor.

Finally, we can fit the model:

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;price=&\quad0.195&space;\times&space;accommodates&space;&plus;0.170&space;\times&space;bathrooms&space;&plus;&space;0.123&space;\times&space;bedrooms\\&space;&&plus;&space;0.028&space;\times&space;beds&space;-&space;0.012&space;\times&space;numbers&space;of&space;reviews&space;&plus;&space;0.001&space;\times&space;host&space;listings&space;count\\&space;&&space;&plus;\sum_{i=0}^na_iI(neighbourhood=neighbourhood_i)&plus;\sum_{i=0}^nb_iI(room&space;type=roomtype_i)&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;price=&\quad0.195&space;\times&space;accommodates&space;&plus;0.170&space;\times&space;bathrooms&space;&plus;&space;0.123&space;\times&space;bedrooms\\&space;&&plus;&space;0.028&space;\times&space;beds&space;-&space;0.012&space;\times&space;numbers&space;of&space;reviews&space;&plus;&space;0.001&space;\times&space;host&space;listings&space;count\\&space;&&space;&plus;\sum_{i=0}^na_iI(neighbourhood=neighbourhood_i)&plus;\sum_{i=0}^nb_iI(room&space;type=roomtype_i)&space;\end{align*}" title="\begin{align*} price=&\quad0.195 \times accommodates +0.170 \times bathrooms + 0.123 \times bedrooms\\ &+ 0.028 \times beds - 0.012 \times numbers of reviews + 0.001 \times host listings count\\ & +\sum_{i=0}^na_iI(neighbourhood=neighbourhood_i)+\sum_{i=0}^nb_iI(room type=roomtype_i) \end{align*}" /></a>

where 𝑎_i is the coefficient of neighbourhood_i, 𝑏_i is the coefficient of roomtype_i and

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=I(x=c)=\left\{{\color{Red}&space;{\color{Green}&space;}}&space;\begin{matrix}&space;1,&space;&&space;x=c&space;\\&space;0,&space;&&space;otherwise&space;\end{matrix}&space;\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?I(x=c)=\left\{{\color{Red}&space;{\color{Green}&space;}}&space;\begin{matrix}&space;1,&space;&&space;x=c&space;\\&space;0,&space;&&space;otherwise&space;\end{matrix}&space;\right." title="I(x=c)=\left\{{\color{Red} {\color{Green} }} \begin{matrix} 1, & x=c \\ 0, & otherwise \end{matrix} \right." /></a>
</p>

### Model accuracy
In order to test the accuracy of our model, we include the definition of error percentage.

**Definition (Error percentage)**: The floating rate of prediction with the true value. For example, if the true price is
100 and the estimation is 120, the error percentage is 0.2.

<img width="380" height="300" src="https://github.com/littlesun0727/Airbnb-Data-Analysis/blob/main/Linear%20Regression/prediction_error.png"><img width="380" height="300" src="https://github.com/littlesun0727/Airbnb-Data-Analysis/blob/main/Linear%20Regression/error_percentage.png">

From Histogram of error percentage, we know that the error percentage of 70% houses in the testing set is below 33%. Therefore, our prediction model is reasonable.

### Importance of all factors
From the coefficients of the model we fit, the influence of all factors on price can be ordered as:

<p align="center">
<img width="500" height="400" src="https://github.com/littlesun0727/Airbnb-Data-Analysis/blob/main/Linear%20Regression/coefficients.png" /> 
</p>

- accommodates > bathrooms > bedrooms > review_scores_rating > security_deposite > cleaning_fee >
beds > number_of_reviews > host_listings_count

