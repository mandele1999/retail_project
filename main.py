# 1. Import Libraries
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template
from IPython.display import display
from datetime import datetime
import os
import argparse
import logging
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

#==================================================================
# 2. Load dataset
def load_data(file_path):
    """Loads dataset from specified path."""
    try: 
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
#==================================================================

# 3. Clean and prepare data
def clean_data(df, save_path=None):
    """Cleans the data by handling missing values, filtering invalid transactions
    and creating new column features"""
    
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0.009)]
    df.dropna(subset=['CustomerID', 'InvoiceDate', 'InvoiceNo'],inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Date'] = df['InvoiceDate'].dt.date
    df['Time'] = df['InvoiceDate'].dt.time
    df['UnitPrice'] = df['UnitPrice'].round(2)
    df['CustomerID'] = df['CustomerID'].astype(int) # Eliminate the decimal
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['TransactionAmount'] = (df['Quantity'] * df['UnitPrice']).round(2)
    # Save cleaned data
    if save_path:
        df.to_pkl(save_path)
    return df
#===================================================================

def get_reference_date(df=None):
    """Get the reference date dynamically from the dataset or via arguments/environment variables."""
    import os
    import argparse
    from datetime import datetime

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run RFM analysis")
    parser.add_argument(
        "--reference_date",
        type=str,
        help="Reference date in YYYY-MM-DD format (e.g., 2024-01-01)",
    )
    args = parser.parse_args()

    # Check command-line argument
    if args.reference_date:
        try:
            return datetime.strptime(args.reference_date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

    # Check environment variable
    ref_date_env = os.getenv("REFERENCE_DATE")
    if ref_date_env:
        try:
            return datetime.strptime(ref_date_env, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Invalid date format in REFERENCE_DATE environment variable. Use YYYY-MM-DD.")

    # Default to dynamic calculation from the dataset
    if df is not None:
        try:
            analysis_date = df['InvoiceDate'].max() + pd.DateOffset(1)
            return analysis_date.date()
        except Exception as e:
            raise ValueError(f"Could not calculate reference date from dataset: {e}")

    # Default to today's date if all else fails (testing purposes)
    print("Using today's date as the default reference date.")
    return datetime.today().date()
#===================================================================

# 4. Calculate RFM metrics
def calculate_rfm(df, reference_date):
    """Calculates Recency, Frequency, and Monetary metrics."""
    
    rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),  
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TransactionAmount', 'sum')  
    ).reset_index()
    return rfm
#====================================================================
# *5.* Transform and score metrics 
def log_transform_rfm(rfm):
    rfm['Recency_log'] = np.log1p(rfm['Recency'])
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    return rfm

def score_rfm_metrics(rfm):
    rfm['Recency_score'] = pd.qcut(rfm['Recency_log'], 5, labels=[5, 4, 3, 2, 1])
    rfm['Frequency_score'] = pd.qcut(rfm['Frequency_log'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['Monetary_score'] = pd.qcut(rfm['Monetary_log'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['RFM_score'] = (rfm['Recency_score'].astype(int) + rfm['Frequency_score'].astype(int) + rfm['Monetary_score'].astype(int))
    return rfm
#====================================================================

# 6. Assign value segments
def assign_value_segments(rfm, segment_labels=None):
    if segment_labels is None:
        segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
    
    rfm['Value Segment'] = pd.qcut(rfm['RFM_score'], q=len(segment_labels), labels=segment_labels)
    return rfm
#====================================================================

# 7. Define customer segments
def define_behavioral_segments(rfm):
    def aggregate_segment(row):
        if row['RFM_score'] >= 13:
            return 'Champions'
        elif row['RFM_score'] >= 10:
            return 'Loyal Customers'
        elif row['RFM_score'] >= 7:
            return 'Potential Loyalists'
        elif row['RFM_score'] >= 5:
            return 'At Risk'
        else:
            return 'Hibernating'
    
    rfm['RFM Customer Segment'] = rfm.apply(aggregate_segment, axis=1)
    return rfm
#======================================================================

# 8. Visualization

def plot_value_segment_distribution(rfm, output_path=None, save=False):
    # Count customers in each value segment
    value_segment_count = rfm['Value Segment'].value_counts().reset_index()
    value_segment_count.columns = ['Value Segment', 'Count']
    
    # Define color palette
    pastel_colors = px.colors.qualitative.Pastel
    
    # Create the bar chart
    fig = px.bar(
        value_segment_count, 
        x='Value Segment', 
        y='Count', 
        color='Value Segment', 
        color_discrete_sequence=pastel_colors,
        title='RFM Value Segment Distribution'
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_title='RFM Value Segment',
        yaxis_title='Count',
        showlegend=False
    )
    
    # Save the plot if output_path is provided
    if save and output_path:
        fig.write_image(output_path, width=800, height=600)
    
    return fig
#----------------------------------------------------------------
def plot_behavioral_segment_distribution(rfm, output_path=None, save=False):
    # Define colors
    pastel_colors = px.colors.qualitative.Pastel
    champions_color = 'rgb(158, 202, 225)'  # Custom color for 'Champions'

    # Count customers in each behavioral segment
    segment_counts = rfm['RFM Customer Segment'].value_counts()
    
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=segment_counts.index, 
            y=segment_counts.values,
            marker=dict(
                color=[
                    champions_color if segment == 'Champions' else pastel_colors[i % len(pastel_colors)]
                    for i, segment in enumerate(segment_counts.index)
                ],
                line=dict(color='rgb(8, 48, 107)', width=1.5)
            ),
            opacity=0.6
        )
    ])
    
    # Update layout
    fig.update_layout(
        title='Comparison of Behavioral Segments',
        xaxis_title='Behavioral Segments',
        yaxis_title='Number of Customers',
        showlegend=False
    )
    
    # Save the plot if output_path is provided
    if save and output_path:
        fig.write_image(output_path, width=800, height=600)
    
    return fig
#----------------------------------------------------------------

def plot_average_rfm_scores(rfm, output_path=None, save=False):
    # Ensure RFM scores are integers
    rfm['Recency_score'] = rfm['Recency_score'].astype(int)
    rfm['Frequency_score'] = rfm['Frequency_score'].astype(int)
    rfm['Monetary_score'] = rfm['Monetary_score'].astype(int)
    
    # Calculate average RFM scores for each segment
    segment_scores = (
        rfm.groupby('RFM Customer Segment')[['Recency_score', 'Frequency_score', 'Monetary_score']]
        .mean()
        .round(1)
        .reset_index()
    )
    
    # Create a grouped bar chart
    fig = go.Figure()
    
    # Add bars for Recency score
    fig.add_trace(go.Bar(
        x=segment_scores['RFM Customer Segment'],
        y=segment_scores['Recency_score'],
        name='Recency Score',
        marker_color='rgb(158,202,225)'
    ))
    
    # Add bars for Frequency score
    fig.add_trace(go.Bar(
        x=segment_scores['RFM Customer Segment'],
        y=segment_scores['Frequency_score'],
        name='Frequency Score',
        marker_color='rgb(94,158,217)'
    ))
    
    # Add bars for Monetary score
    fig.add_trace(go.Bar(
        x=segment_scores['RFM Customer Segment'],
        y=segment_scores['Monetary_score'],
        name='Monetary Score',
        marker_color='rgb(32,102,148)'
    ))
    
    # Update layout
    fig.update_layout(
        title='Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores',
        xaxis_title='RFM Segments',
        yaxis_title='Score',
        barmode='group',
        showlegend=True
    )
    
    # Save the plot if output_path is provided
    if save and output_path:
        fig.write_image(output_path, width=800, height=600)
    
    return fig
#-----------------------------------------------------------------
def plot_revenue_contribution(rfm, output_path=None, save=False):
    # Calculate total revenue per segment
    segment_revenue = (
        rfm.groupby('RFM Customer Segment')['Monetary']
        .sum()
        .reset_index()
        .rename(columns={'Monetary': 'Total Revenue'})
    )
    
    # Add a percentage contribution column
    total_revenue = segment_revenue['Total Revenue'].sum()
    segment_revenue['Revenue Percentage'] = (
        (segment_revenue['Total Revenue'] / total_revenue) * 100
    ).round(2)
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        segment_revenue['Total Revenue'],
        labels=segment_revenue['RFM Customer Segment'],
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('Blues', len(segment_revenue))
    )
    
    # Add a title
    ax.set_title('Revenue Contribution by Customer Segment', fontsize=16)
    plt.tight_layout()
    
    # Save the plot if output_path is provided
    if save and output_path:
        fig.savefig(output_path, dpi=300)
    
    return fig
#======================================================================
# Generate reports

def save_rfm_results(rfm, output_path):
    """Exports RFM analysis results to a Pickle file."""
    rfm.to_pickle(output_path, index=False)

def save_summary_table(data, output_path):
    """Saves a summary table to a text file."""
    with open(output_path, "w") as file:
        file.write(data)
#======================================================================
# Main workflow
def main(data_path, output_pkl, summary_txt):
    # Step 1: Load data
    df = load_data(data_path)
    if df is None:
        logging.error("Failed to load data. Exiting.")
        return
    # Step 2: Clean data
    df = clean_data(df)

    # Get reference date dynamically or via configurations
    reference_date = get_reference_date(df)
    logging.info(f"Reference date is: {reference_date}")

    # Step 3: Calculate RFM metrics
    rfm = calculate_rfm(df, reference_date)

    # Step 4: Transform and Score RFM metrics
    rfm = log_transform_rfm(rfm)

    # Step 5: Score RFM metrics
    rfm = score_rfm_metrics(rfm)

    # Step 6: Segment customers on value basis
    rfm = assign_value_segments(rfm)

    # Step 7: Segment customers based on purchasing behavior
    rfm = define_behavioral_segments(rfm)

    # Step 8: Visualize data
    output_paths = {
    "value_segment": "../images/rfm_value_segment_dist.png",
    "behavioral_segment": "../images/rfm_segments_comparison.png",
    "rfm_comparisons": "../images/rfm_comparisons.png",
    "revenue_contribution": "../images/revenue_contribution.png",
    }

    # Usage
    fig_segment_dist = plot_value_segment_distribution(rfm, output_paths["value_segment"])
    fig_behavioral_dist = plot_behavioral_segment_distribution(rfm, output_paths["behavioral_segment"])
    fig_avg_rfm_scores = plot_average_rfm_scores(rfm, output_paths["rfm_comparisons"])
    fig_revenue_contribution = plot_revenue_contribution(rfm, output_paths["revenue_contribution"])

    # Step 9: Save results
    save_rfm_results(rfm, output_pkl)

    # Step 10: Save summary
    summary_data = [
        ["Champions", "25.13%", "70.19%"],
        ["Loyal Customers", "27.14%", "15.81%"],
        ["Potential Loyalists", "29.35%", "9.90%"],
        ["At Risk", "20.45%", "2.92%"],
        ["Hibernating", "14.68%", "1.18%"]
    ]
    summary_table = tabulate(summary_data, headers=["Segment", "Customer %", "Revenue %"], tablefmt="grid")
    save_summary_table(summary_table, summary_txt)
#======================================================================
# Commande execution
if __name__ == "__main__":
    reference_date = datetime(2011, 12, 10)  # Update as needed
    main(
        data_path="../data/raw/online_retail.csv",
        output_pkl="../data/processed/rfm_table.pkl",
        summary_txt="../images/segment_comparison_table.txt",
        reference_date=reference_date
    )
