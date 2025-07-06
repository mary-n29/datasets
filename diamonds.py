import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/mary-n29/datasets/refs/heads/main/Diamonds%20export%202025-06-15%2012-53-58.csv"
df = pd.read_csv(url)
pd.set_option('display.float_format', '{:.2f}'.format)

#*****************************************************************
#data cleaning

# Drop all rows with any missing values
df = df.dropna()

# Confirm no missing values remain
#print("\nMissing values per column after dropping:")
#print(df.isna().sum())


# Check how many rows have 0 in critical columns before dropping
zero_counts = (df[['x', 'y', 'z', 'carat', 'price']] == 0).sum()
print("Count of zero values in critical columns before dropping:")
print(zero_counts)

# Drop rows where any of the critical columns are 0
df = df[(df[['x', 'y', 'z', 'carat', 'price']] != 0).all(axis=1)]

# Confirm zeros are removed
zero_counts_after = (df[['x', 'y', 'z', 'carat', 'price']] == 0).sum()
print("\nCount of zero values in critical columns after dropping:")
print(zero_counts_after)

# Confirm new shape
print(f"\nNew dataset shape after dropping zero-value rows: {df.shape}")

#*****************************************************************
# feature engineering:   create 'volume' column, 'diameter' column
df['diameter'] = (df['x'] + df['y']) / 2
df['volume'] = (df['x'] * df['y'] * df['z']).round(2)

zero_volumes = (df['volume'] == 0).sum()
print(f"\nNumber of zero volume entries: {zero_volumes}")

# Calculate quantiles for volume
quantiles = df['volume'].quantile([0.33, 0.75])
q33 = quantiles.loc[0.33]
q75 = quantiles.loc[0.75]

print(f"33rd percentile (q33): {q33:.2f}")
print(f"75th percentile (q75): {q75:.2f}")

# Define bins using quantiles
bins = [0, q33, q75, float('inf')]
labels = ['Small', 'Medium', 'Large']

# Categorize diamonds based on volume
df['size_tier'] = pd.cut(
    df['volume'],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True
)


# Confirm dataset structure, ranges, and distribution
print(df.info())

print("\nDescriptive statistics (2-decimal precision):")
print(df.describe().round(2))

print("\nSize tier distribution:")
print(df['size_tier'].value_counts())

print(df.head())

#*****************************************************************
#outliers

# ensure numeric conversion to handle hidden strings
cols_to_clean = ['carat', 'price', 'depth', 'table', 'x', 'y', 'z', 'volume']
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Record shape before outlier removal
initial_shape = df.shape

# IQR-based outlier detection and removal
for col in cols_to_clean:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Record shape after outlier removal
final_shape = df.shape

# View results
print(f"Initial dataset shape before outlier removal: {initial_shape}")
print(f"Final dataset shape after outlier removal: {final_shape}")
print("\nFirst few rows of the cleaned dataset:")
print(df.head())

print("\nDescriptive statistics after outlier removal:")
print(df.describe().round(2))


#*****************************************************************
#pearson correlation

# Select numerical features for correlation analysis
numerical_features = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z', 'volume', 'diameter']

# Calculate the Pearson correlation matrix
corr_matrix = df[numerical_features].corr(method='pearson').round(2)

# Plot heatmap of the Pearson correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            linewidths=0.5,
            fmt='.2f')
plt.title('Heatmap of Pearson Correlation Coefficients for Diamond Features', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Generate scatter plot matrix (pairplot) to visualize feature relationships
pairplot = sns.pairplot(df[numerical_features],
                        corner=True,
                        plot_kws={'alpha': 0.3, 's': 10},
                        diag_kws={'bins': 20},
                        height=1.8)
pairplot.fig.suptitle('Scatter Plot Matrix of Diamond Features', fontsize=16, fontweight='bold', y=1.02)
plt.show()

#explanation table
from prettytable import PrettyTable

# Create and define table structure
table = PrettyTable()
table.field_names = ["Feature 1", "Feature 2", "Correlation", "Observation"]

# Populate with your dataset's key correlation insights
table.add_row(["carat", "price", "0.93", "Strong positive; larger diamonds cost more"])
table.add_row(["volume", "price", "0.93", "Strong positive; volume influences price similarly to carat"])
table.add_row(["carat", "volume", "0.99", "Nearly perfect; both measure size, indicating redundancy"])
table.add_row(["depth", "price", "0.02", "No significant correlation; depth has little impact on price"])
table.add_row(["table", "price", "0.14", "Very weak correlation; minor influence on price"])
table.add_row(["depth", "table", "-0.23", "Weak negative; slight trade-off in cut design"])

# Print the table for direct copy into your report
print(table)

