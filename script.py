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
# 2. Load Dataset
def load_data(file_path):
    """Loads dataset from specified path."""
    try: 
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

#==================================================================
# 3. Clean and Prepare Data
def clean_data(df, save_path=None):
    """Cleans the dataset by handling missing values, filtering invalid transactions,
    and creating new column features."""
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0.009)]
    df.dropna(subset=['CustomerID', 'InvoiceDate', 'InvoiceNo'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Date'] = df['InvoiceDate'].dt.date
    df['Time'] = df['InvoiceDate'].dt.time
    df['UnitPrice'] = df['UnitPrice'].round(2)
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['TransactionAmount'] = (df['Quantity'] * df['UnitPrice']).round(2)
    if save_path:
        df.to_pickle(save_path)
    return df

#==================================================================
# 4. Get Reference Date
def get_reference_date(df=None):
    """Determines the reference date dynamically or via arguments/environment variables."""
    parser = argparse.ArgumentParser(description="Run RFM analysis")
    parser.add_argument("--reference_date", type=str, help="Reference date in YYYY-MM-DD format")
    args = parser.parse_args()
    
    if args.reference_date:
        try:
            return datetime.strptime(args.reference_date, "%Y-%m-%d").date()
        except ValueError:
            logging.error("Invalid date format. Use YYYY-MM-DD.")
            raise

    ref_date_env = os.getenv("REFERENCE_DATE")
    if ref_date_env:
        try:
            return datetime.strptime(ref_date_env, "%Y-%m-%d").date()
        except ValueError:
            logging.error("Invalid environment variable date format. Use YYYY-MM-DD.")
            raise

    if df is not None:
        try:
            return (df['InvoiceDate'].max() + pd.DateOffset(1)).date()
        except Exception as e:
            logging.error(f"Failed to calculate reference date: {e}")
            raise

    logging.warning("Defaulting to today's date.")
    return datetime.today().date()

#==================================================================
# 5. Calculate RFM Metrics
def calculate_rfm(df, reference_date):
    """Calculates Recency, Frequency, and Monetary metrics."""
    return df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (pd.Timestamp(reference_date) - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TransactionAmount', 'sum')
    ).reset_index()

#==================================================================
# 6. Transform and Score RFM Metrics
def log_transform_rfm(rfm):
    """Applies log transformation to RFM metrics."""
    for col in ['Recency', 'Frequency', 'Monetary']:
        rfm[f'{col}_log'] = np.log1p(rfm[col])
    return rfm

def score_rfm_metrics(rfm):
    """Assigns scores to RFM metrics."""
    rfm['Recency_score'] = pd.qcut(rfm['Recency_log'], 5, labels=[5, 4, 3, 2, 1])
    rfm['Frequency_score'] = pd.qcut(rfm['Frequency_log'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['Monetary_score'] = pd.qcut(rfm['Monetary_log'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['RFM_score'] = rfm[['Recency_score', 'Frequency_score', 'Monetary_score']].sum(axis=1).astype(int)
    return rfm

#==================================================================
# 7. Assign Customer Segments
def assign_value_segments(rfm, segment_labels=None):
    """Assigns value-based segments."""
    segment_labels = segment_labels or ['Low-Value', 'Mid-Value', 'High-Value']
    rfm['Value Segment'] = pd.qcut(rfm['RFM_score'], q=len(segment_labels), labels=segment_labels)
    return rfm

def define_behavioral_segments(rfm):
    """Assigns behavioral segments based on RFM score."""
    def classify(row):
        if row['RFM_score'] >= 13:
            return 'Champions'
        elif row['RFM_score'] >= 10:
            return 'Loyal Customers'
        elif row['RFM_score'] >= 7:
            return 'Potential Loyalists'
        elif row['RFM_score'] >= 5:
            return 'At Risk'
        return 'Hibernating'
    rfm['RFM Customer Segment'] = rfm.apply(classify, axis=1)
    return rfm

#==================================================================
# 8. Visualization
# Includes all plotting functions as provided earlier, unchanged.
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



#==================================================================
# 9. Save Results
def save_rfm_results(rfm, output_path):
    """Saves RFM analysis results to a file."""
    rfm.to_pickle(output_path)

def save_summary_table(data, output_path):
    """Saves summary table to a text file."""
    with open(output_path, "w") as file:
        file.write(data)
#==================================================================
# 10. Generate HTML Report
def generate_report(rfm, comparison_table, output_html="notebooks/rfm_analysis_report.html"):
    """
    Generates an HTML report for RFM analysis.

    Parameters:
        rfm (DataFrame): The RFM analysis data.
        comparison_table (list of dict): Preprocessed customer comparison data for the report.
        output_html (str): Path to save the rendered HTML report.

    Returns:
        None
    """
    try:
        # Load the HTML template
        with open("report_template.html", "r") as file:
            html_template = file.read()

        template = Template(html_template)

        # Define your data for the report
        overview = """
        This report presents the findings of an RFM (Recency, Frequency, Monetary) analysis conducted to evaluate customer behavior and identify actionable insights for business growth.
        Customers were segmented into behavioral and value-based groups, with visualizations highlighting key patterns and opportunities for strategic action.

        The analysis includes revenue contributions, customer distributions across segments, and specific recommendations for engaging and retaining different customer groups.
        """
        insights = [
            "Champions contribute the highest revenue at 70.19% despite being 20% of the customer base.",
            "Hibernating customers generate only 1.18% of revenue, highlighting low engagement.",
            "Potential Loyalists and Loyal Customers together contribute 25.71% of the revenue, indicating opportunities for growth.",
            "At Risk customers contribute 2.92% of revenue, warranting strategies to re-engage them."
        ]
        # Set relative paths for images
        base_path = os.path.dirname(output_html)  # Path where the report is being saved
        image_paths = {
            "plot_revenue": os.path.join(base_path, "revenue_contribution.png"),
            "plot_segments": os.path.join(base_path, "rfm_segments_comparison.png"),
            "plot_value_segments": os.path.join(base_path, "rfm_value_segment_dist.png"),
            "plot_rfm_comparison": os.path.join(base_path, "rfm_comparisons.png")
        }

        context = {
    "overview": overview,
    "segment_revenue": [
        {"RFM Customer Segment": "Champions", "Total Revenue": 6255336.53, "Revenue Percentage": 70.19},
        {"RFM Customer Segment": "Loyal Customers", "Total Revenue": 1408629.91, "Revenue Percentage": 15.81},
        {"RFM Customer Segment": "Potential Loyalists", "Total Revenue": 882612.66, "Revenue Percentage": 9.90},
        {"RFM Customer Segment": "At Risk", "Total Revenue": 259877.89, "Revenue Percentage": 2.92},
        {"RFM Customer Segment": "Hibernating", "Total Revenue": 104950.91, "Revenue Percentage": 1.18}
    ],
    "plot_revenue": "../images/revenue_contribution.png",
    "plot_segments": "../images/rfm_segments_comparison.png",
    "plot_value_segments": "../images/rfm_value_segment_dist.png",
    "plot_rfm_comparison": "../images/rfm_comparisons.png",
    "customer_table": comparison_table,
    "insights": insights,

    "recommendations": [
        "Strengthening relationships with champions: Focus on retention strategies like exclusive loyalty programs, early access to products, or premium services. Ensure consistent and personalized communicatio  to maintain their satisfaction and speding levels.",
        "Nurture loyal customers and potential loyalists: Create targeted campaigns to increase engagement, such as offering tiered rewards for higher spending or incentivizing frequent purchases. Educate these customers about additional products/services to encourage upselling and cross-selling",
        "Re-engage At Risk customers: Deploy win-back campaigns, including special offers, personalized outreach, or feedback collection to understand their disengagement. Focus on reactivating high-value customers within this segment.",
        "Optimize efforts for hibernating customers: Periodic reminders or seasonal promotions can re-engage this group, but prioritize resources on higher-value segments.",
        "Monitor and track performance: Regularly update RFM scores and revenue contributions to identify changes in customer behavior. Use these insights to refine marketing and operational strategies",
        # Add more recommendations
        ]
        }

        # Render the template with the context data
        rendered_html = template.render(context)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_html), exist_ok=True)

        # Save the rendered HTML to a file
        with open(output_html, "w") as report_file:
            report_file.write(rendered_html)

        logging.info(f"Report generated successfully: {output_html}")

    except FileNotFoundError as e:
        logging.error(f"Template file not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred while generating the report: {e}")


#==================================================================
# 10. Main Workflow
def main(data_path, output_pkl, summary_txt, report_html="rfm_analysis_report.html"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df = load_data(data_path)
    if df is None:
        return
    df = clean_data(df)
    reference_date = get_reference_date(df)
    logging.info(f"Reference date: {reference_date}")
    
    rfm = calculate_rfm(df, reference_date)
    rfm = log_transform_rfm(rfm)
    rfm = score_rfm_metrics(rfm)
    rfm = assign_value_segments(rfm)
    rfm = define_behavioral_segments(rfm)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)

    save_rfm_results(rfm, output_pkl)

    summary_data = [
        ["Champions", "25.13%", "70.19%"],
        ["Loyal Customers", "27.14%", "15.81%"],
        ["Potential Loyalists", "29.35%", "9.90%"],
        ["At Risk", "20.45%", "2.92%"],
        ["Hibernating", "14.68%", "1.18%"]
        ]
    headers = ["Segment", "Customer Proportion", "Revenue Contribution"]
    summary_table = tabulate(summary_data, headers=headers, tablefmt="grid")
    save_summary_table(summary_table, summary_txt)

# Paths for saving plots
    output_paths = {
        "value_segment": "images/rfm_value_segment_dist.png",
        "behavioral_segment": "images/rfm_segments_comparison.png",
        "rfm_comparisons": "images/rfm_comparisons.png",
        "revenue_contribution": "images/revenue_contribution.png",
    }

    # Generate and save visualizations
    logging.info("Generating and saving plots...")
    plot_value_segment_distribution(rfm, output_paths["value_segment"], save=True)
    plot_behavioral_segment_distribution(rfm, output_paths["behavioral_segment"], save=True)
    plot_average_rfm_scores(rfm, output_paths["rfm_comparisons"], save=True)
    plot_revenue_contribution(rfm, output_paths["revenue_contribution"], save=True)
    logging.info("Plots saved successfully.")
    
    generate_report(rfm=rfm, comparison_table=summary_table)

    logging.info(f"Report generated: {report_html}")


    logging.info("Workflow completed successfully.")

if __name__ == "__main__":
    main(
        data_path="data/raw/online_retail.csv",
        output_pkl="../data/processed/rfm_table.pkl",
        summary_txt="images/segment_comparison_table.txt",
        report_html="notebooks/rfm_analysis_report.html"
    )
