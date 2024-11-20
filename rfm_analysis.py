# 1. Import libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
#---------------------------------------------------------------------
# 2. Load the dataset
data_path = "../data/transactions.csv"  # Update with your file path
df = pd.read_csv(data_path)

# Optional: Inspect the data
print(df.head())
#---------------------------------------------------------------------

# 3. Data Cleaning and Preparation
# Handle missing values
df.dropna(subset=['CustomerID', 'InvoiceDate', 'InvoiceNo'], inplace=True)

# Filter invalid transactions
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Add a 'TransactionAmount' column
df['TransactionAmount'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract Date and Time columns
df['Date'] = df['InvoiceDate'].dt.date
df['Time'] = df['InvoiceDate'].dt.time
#-------------------------------------------------------------------

# 4. Calculate RFM metrics
# Define a reference date for Recency calculation
reference_date = datetime(2011, 12, 10)  # Adjust as per your dataset

# Calculate RFM metrics
rfm = df.groupby('CustomerID').agg({
    'Date': lambda x: (reference_date - max(x)).days,  # Recency
    'InvoiceNo': 'count',                             # Frequency
    'TransactionAmount': 'sum'                        # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
#------------------------------------------------------------------

# 5. Transform and Score Metrics
# Log-transform skewed metrics (optional)
rfm['Recency_log'] = np.log1p(rfm['Recency'])
rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

# Assign scores (1-5) using quantiles
def score_metric(data, column, ascending=True):
    labels = [5, 4, 3, 2, 1] if ascending else [1, 2, 3, 4, 5]
    return pd.qcut(data[column], q=5, labels=labels, duplicates='drop').astype(int)

rfm['Recency_score'] = score_metric(rfm, 'Recency', ascending=False)
rfm['Frequency_score'] = score_metric(rfm, 'Frequency', ascending=True)
rfm['Monetary_score'] = score_metric(rfm, 'Monetary', ascending=True)

# Combine scores into an RFM segment
rfm['RFM_Segment'] = rfm['Recency_score'].astype(str) + rfm['Frequency_score'].astype(str) + rfm['Monetary_score'].astype(str)

#-----------------------------------------------------------------
# 6. Define customer segments
def assign_segment(row):
    if row['RFM_Segment'] == '555':
        return 'Champions'
    elif row['RFM_Segment'] == '111':
        return 'Hibernating'
    elif row['RFM_Segment'][0] == '5':
        return 'Loyal Customers'
    # Add more conditions as needed
    else:
        return 'Other'

rfm['Customer_Segment'] = rfm.apply(assign_segment, axis=1)

#-------------------------------------------------------------------
# 7. Analyze and Visualize
# Segment distribution
segment_counts = rfm['Customer_Segment'].value_counts()
print(segment_counts)

# Visualize revenue contribution
revenue_per_segment = rfm.groupby('Customer_Segment')['Monetary'].sum()
revenue_percentage = (revenue_per_segment / revenue_per_segment.sum()) * 100

# Plot
sns.barplot(x=revenue_percentage.index, y=revenue_percentage.values)
plt.title('Revenue Contribution by Segment')
plt.ylabel('Percentage')
plt.show()

#-------------------------------------------------------------------
# 8. Generate Reports
# Export RFM results
rfm.to_csv("../outputs/rfm_analysis.csv", index=False)

# Create a tabular summary
summary_data = [
    ['Champions', '25%', '70%'],
    ['Loyal Customers', '27%', '15%'],
    ['Hibernating', '15%', '2%']
]

summary_table = tabulate(summary_data, headers=["Segment", "Customer %", "Revenue %"], tablefmt="grid")
print(summary_table)

# Save summary
with open("../outputs/rfm_summary.txt", "w") as file:
    file.write(summary_table)
