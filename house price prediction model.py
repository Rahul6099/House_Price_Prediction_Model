#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[1]:


# Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries for regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Data visualization settings (optional)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

# Ignore warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility (optional)
np.random.seed(0)


# # LOADING FILES AND DATA

# In[2]:


file_path = 'C:/Users/engin/Downloads/house.csv'


# In[3]:


# Read the CSV file into a DataFrame
house = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify the import
house.head()


# # DATA ANALYSIS

# In[4]:


house.shape


# In[5]:


house.info()


# In[6]:


house.isnull().sum()


# In[7]:


house.isnull().sum()


# In[8]:


house.describe()


# In[9]:


house['houseprice'].value_counts()


# In[ ]:





# # DATA VISUALIZATION 

# # PAIRWISE HISTOGRAM

# In[10]:


house.hist(bins=100, figsize=(25,25))
plt.show()


# # COUNT PLOT

# In[11]:


plt.figure(figsize=(10, 7))

# Create the countplot
ax = sns.countplot(data=house, x='houseprice')

# Set the y-axis limits to a maximum of 200
ax.set(ylim=(0, 200))

# Show the plot
plt.show()


# # SCATTER PLOT

# In[12]:


import matplotlib.pyplot as plt

plt.scatter(house['rooms'], house['houseprice'])


# # Heatmap of Correlation

# In[13]:


plt.figure(figsize=(10, 7))
sns.heatmap(house.corr(), annot=True)
plt.title('Correlation Between the Columns')
plt.show()


# In[14]:


import matplotlib.pyplot as plt

# Replace 'variable' with the actual column name from your 'house' DataFrame
plt.hist(house['houseprice'], bins=20)
plt.xlabel('age')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()


# In[15]:


import matplotlib.pyplot as plt

plt.plot(house['income'], house['houseprice'])
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Line Plot')
plt.show()


# In[16]:


house.corr()['houseprice'].sort_values()


# # BARPLOT

# In[17]:


sns.barplot(x=house['houseprice'], y=house['income'])
plt.title('Bar Plot of houseprice vs income')
plt.show()


# # Paiwise Plot of House Price vs. Number of Rooms

# In[18]:


sns.pairplot(house, vars=['rooms', 'population', 'income', 'houseprice'], diag_kind='kde')
plt.show()



# # Scatter Plot of House Price vs. Number of Rooms

# In[19]:


plt.scatter(house ['rooms'], house ['houseprice'], alpha=0.5, color='green')
plt.title('House Price vs. Number of Rooms')
plt.xlabel('Number of Rooms')
plt.ylabel('House Price')
plt.show()


# # HISTOGRAM

# In[20]:


plt.hist(house ['houseprice'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of House Prices')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.show()


# In[21]:


correlation_matrix = house.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# # BOXPLOT

# In[22]:


import seaborn as sns

sns.boxplot(x='age', data=house, color='red')
plt.title('Distribution of House Age')
plt.xlabel('House Age')
plt.show() 


# In[23]:


count_values = house['houseprice'].value_counts()
print(count_values)


# # REGRESSION MODEL DEVELOPMENT 

# # Split the data into features (X) and the target (y)

# In[35]:


X = house.drop('houseprice', axis=1)  # Features (exclude the target column)
y = house['houseprice']  # Target variable


# # Split the data into training and testing sets

# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Create a Linear Regression model and fit it to the training data

# In[37]:


model = LinearRegression()
model.fit(X_train, y_train)


# # Make predictions on the test set

# In[38]:


y_pred = model.predict(X_test)


# # confusion matrix

# In[28]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Actual labels (ground truth)
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]

# Predicted labels (model's predictions)
y_pred = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create a heatmap for visualization
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# # Evaluate the model's performance

# In[39]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# # make predictions on new data by providing the features of a house. For example

# In[32]:


new_house_features = [-122.26, 37.84, 30, 1400, 600, 220, 4.0]
predicted_price = model.predict([new_house_features])
print(f"Predicted Price: {predicted_price[0]}")


# In[ ]:




