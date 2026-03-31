## Linear Regression from Scratch(Numpy Only)
Implementation of Linear Regression using Numpy only and Gradient Descent For Optimization. The Dataset Which is Used for the Linear Regression is Housing Dataset.

## Dataset
The housing Dataset link : https://www.kaggle.com/datasets/camnugent/california-housing-prices

## Project Setup
# Create a virtual environment
python -m venv venv

## Activate the environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

## Installation of Dependencies
pip install -r requirements.txt

# Run The Program
python main.py

## Approach
1. Loaded The California Housing Dataset
2. Splitting The training and testing Data
3. Standardizarion of The training and testing Data so that the model does not explode with big numbers
4. Then Writing the functions and formulas to find predicted value,gradient, loss, R-squared
5. Training and Testing Process
## Learnings
1. Learnt About the Linear Regression model from Scratch
2. Learnt The importance of Standardization
3. Learnt the Concept of Mean Squared Error
## Result
1. Learned Equation => y= 2547286293.00*X_train_norm + 2706264130.45
2. R-Squared => Train R²=0.4758 Test R²=0.4643
## Difficulty and Resolution
1. Difficulty - The Model was not able to train the data as house sizes (thousands) were too large for the weights. 
2. Resolution - Used Normalization to bring all values between -2 and 2.
