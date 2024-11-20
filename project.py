import pandas as pd
from sklearn.model_selection import train_test_split


# Load train and test datasets
train_data = pd.read_csv("./Dataset/yelp_review_fine-grained_5_classes_csv/train.csv")
test_data = pd.read_csv("./Dataset/yelp_review_fine-grained_5_classes_csv/test.csv")

# Rename the columns in both train and test datasets to more understandable rating, review
train_data.rename(columns={"class_index": "rating", "review_text": "review"}, inplace=True)
test_data.rename(columns={"class_index": "rating", "review_text": "review"}, inplace=True)

# Stratified sampling for creating subsets
train_subset, _ = train_test_split(train_data, train_size=10000, stratify=train_data['rating'], random_state=42)
test_subset, _ = train_test_split(test_data, train_size=2000, stratify=test_data['rating'], random_state=42)

# Reset the index for both train and test subsets
train_subset.reset_index(drop=True, inplace=True)
test_subset.reset_index(drop=True, inplace=True)

# print("Train data preview:")
# print(train_data.head())
# print(train_data.shape)
# print(train_data.columns)

# print("\n Test data preview:")
# print(test_data.head())
# print(test_data.shape)
# print(test_data.columns)

print("Train data preview:")
print(train_subset.head())
print(train_subset.shape)
print(train_subset.columns)
print(train_subset['rating'].value_counts())

print("\n Test data preview:")
print(test_subset.head())
print(test_subset.shape)
print(test_subset.columns)
print(test_subset['rating'].value_counts())


