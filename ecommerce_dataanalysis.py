import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import numpy as np


file_path = 'E:\Program\DAproject\maindataset.csv'
df = pd.read_csv(file_path)


print(df.head(5))  
print(df.tail(5))  

# Shuffle the dataset to avoid any ordering bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

Q1 = df['selling price'].quantile(0.25)
Q3 = df['selling price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['selling price'] >= (Q1 - 1.5 * IQR)) & (df['selling price'] <= (Q3 + 1.5 * IQR))]

le_season = LabelEncoder()
le_brand = LabelEncoder()
df['season_encoded'] = le_season.fit_transform(df['season'])
df['brand_encoded'] = le_brand.fit_transform(df['brand'])


df['rating_to_price_ratio'] = df['average rating'] / df['selling price']


X = df[['season_encoded', 'brand_encoded', 'average rating', 'rating_to_price_ratio']]
y = df['selling price']

np.random.seed(42)
y = y + np.random.normal(0, 10, size=y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5, max_samples=0.8)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Overall accuracy rate for the entire dataset
df['predicted_sales'] = rf.predict(X)
df['absolute_error'] = abs(df['selling price'] - df['predicted_sales'])
df['accuracy_rate'] = 100 - (df['absolute_error'] / df['selling price'] * 100)
overall_accuracy = df['accuracy_rate'].mean()
print(f"\nOverall Accuracy Rate: {overall_accuracy:.2f}%")

# Get the top 10 products sold in each season
top_products_by_season = df.groupby(['season', 'title'])['selling price'].sum().reset_index()
top_products_by_season = top_products_by_season.sort_values(['season', 'selling price'], ascending=[True, False])
top_10_per_season = top_products_by_season.groupby('season').head(10)

# Print the top 10 products in each season
print("\nTop 10 Products Sold in Each Season:")
print(top_10_per_season[['season', 'title', 'selling price']])

# Visualization of Top 10 Products Sold in Each Season
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_per_season, x='title', y='selling price', hue='season', dodge=False)
plt.xticks(rotation=90)
plt.title('Top 10 Products Sold in Each Season')
plt.xlabel('Product Title')
plt.ylabel('Total Sales')
plt.show()

# Box plot of Selling Price by Season
plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='selling price', data=df)
plt.title('Selling Price Distribution by Season')
plt.xlabel('Season')
plt.ylabel('Selling Price')
plt.grid(True)
plt.show()

# Visualization of Actual vs Predicted Sales
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sns.pairplot(comparison_df)
plt.title('Pairplot of Actual vs Predicted Sales')
plt.show()

# Histogram of the accuracy rate for each sale
plt.figure(figsize=(8, 6))
sns.histplot(df['accuracy_rate'], bins=20, kde=True, color='green')
plt.title('Distribution of Accuracy Rate for Each Sale')
plt.xlabel('Accuracy Rate (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()