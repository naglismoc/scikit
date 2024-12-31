import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


mpg = sns.load_dataset('mpg')
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.max_rows', None)
# print(mpg.info())

mpg.loc[32, 'horsepower'] = 75
mpg.loc[126, 'horsepower'] = 105
mpg.loc[330, 'horsepower'] = 51
mpg.loc[336, 'horsepower'] = 120
mpg.loc[354, 'horsepower'] = 73
mpg.loc[374, 'horsepower'] = 82

# print(mpg[mpg['horsepower'].isnull()])


dumies = pd.get_dummies(mpg['origin'])
# print(mpg.head())
# print(mpg['origin'].head())
# print(dumies.head())

# print(mpg)

data = pd.concat([mpg,dumies], axis=1)

# print(data.tail())

# print(mpg['origin'].unique())

data.drop(['origin','name'],axis=1, inplace=True)
# print(data.head())


# print(data.columns)

x = data.drop('mpg',axis=1)
# print(data.head())

y = data['mpg']
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)


reg = LinearRegression().fit(x_train,y_train)

pred = reg.predict(x_test)
# print(pred)

guesses = pd.Series(data=pred, name='Guess')
# print(guesses)

res = pd.concat([y_test.reset_index(drop=True),guesses],axis=1)

# print(res)
#
# print(reg.score(x_train,y_train)*100)
# print(reg.score(x_test,y_test)*100)


print(reg.coef_)

# plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)], color='red')
# plt.scatter(y_test,guesses)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs Predicted')
# plt.show()

# plt.plot(y_test.values, label='Actual', color='blue',linewidth=2)
# plt.plot(guesses, label='Predicted', color='red',linewidth=2)
#
# plt.xlabel('Index')
# plt.ylabel('mpg')
# plt.title('Actual vs Predicted')
# plt.legend()
# plt.show()


plt.plot(y_test.values - guesses, label='Difference', color='green', linewidth=2)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('Difference: Actual vs Predicted')
plt.legend()
plt.show()
