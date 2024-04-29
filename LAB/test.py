import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

url = '/Users/luisinfanten/Desktop/IE/Classes/First-Year/Second-Semester/Simulating and Modelling/Models/Notebooks/LAB/EcomExpense.csv'
data = pd.read_csv(url, index_col=0)

data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
data = pd.get_dummies(data, columns=['CityTier'], drop_first=False)

model_formula = 'TotalSpend ~ Age + Items + MonthlyIncome + TransactionTime + Record + Gender_Male + CityTier_1 + CityTier_2 + CityTier_3'
model = smf.ols(formula=model_formula, data=data).fit()
model.summary()

