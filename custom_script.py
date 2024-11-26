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
    """
    Loads dataset from the specified path and performs basic validation.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        pd.DataFrame or None: Loaded dataset or None if an error occurs.
    """
    try:
        # Confirm the file exists
        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            return None

        # Load the dataset
        df = pd.read_csv(file_path)

        # Check if the DataFrame is empty
        if df.empty:
            logging.warning(f"The dataset is empty: {file_path}")
            return None

        logging.info(f"Dataset loaded successfully: {file_path}")
        return df

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing the file: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {type(e).__name__}: {e}")
        return None

#==================================================================
# 3. Clean and Prepare Data
def clean_data(df, save_path=None):
    """
    Cleans the dataset by handling missing values, filtering invalid transactions,
    and creating new column features.
    
    Args:
        df (pd.DataFrame): Input dataset to clean.
        save_path (str, optional): Path to save the cleaned DataFrame as a pickle file.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    try:
        # Filter for valid transactions
        logging.info("Filtering transactions with valid Quantity and UnitPrice...")
        original_count = len(df)
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0.009)]
        filtered_count = len(df)
        logging.info(f"Rows filtered: {original_count - filtered_count}")

        # Drop rows with missing critical information
        logging.info("Dropping rows with missing values in critical columns...")
        df.dropna(subset=['CustomerID', 'InvoiceDate', 'InvoiceNo'], inplace=True)

        # Convert InvoiceDate to datetime
        logging.info("Converting InvoiceDate to datetime...")
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

        # Add derived columns
        logging.info("Adding derived columns...")
        df['Date'] = df['InvoiceDate'].dt.date
        df['Time'] = df['InvoiceDate'].dt.time
        df['UnitPrice'] = df['UnitPrice'].round(2)
        df['CustomerID'] = df['CustomerID'].astype(str)  # Safe conversion to string
        df['TransactionAmount'] = (df['Quantity'] * df['UnitPrice']).round(2)

        # Save the cleaned DataFrame, if specified
        if save_path:
            logging.info(f"Saving cleaned data to {save_path}...")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_pickle(save_path)

        logging.info("Data cleaning completed successfully.")
        return df

    except KeyError as e:
        logging.error(f"Missing column during cleaning: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during cleaning: {type(e).__name__}: {e}")
        return None


#==================================================================
# 4. Get Reference Date
def get_reference_date(df=None, default_to_today=True):
    """
    Determines the reference date dynamically from command-line arguments, 
    environment variables, or the dataset.

    Args:
        df (pd.DataFrame, optional): Dataset to derive the reference date if no other sources are provided.
        default_to_today (bool, optional): Whether to default to today's date if no valid date is found.

    Returns:
        datetime.date: The reference date.

    Raises:
        ValueError: If no valid reference date can be determined.
    """
    try:
        # Command-line argument
        parser = argparse.ArgumentParser(description="Run RFM analysis")
        parser.add_argument("--reference_date", type=str, help="Reference date in YYYY-MM-DD format")
        args, _ = parser.parse_known_args()  # Avoid parsing errors in interactive environments
        if args.reference_date:
            logging.info("Using reference date from command-line arguments.")
            try:
                return datetime.strptime(args.reference_date, "%Y-%m-%d").date()
            except ValueError:
                logging.error("Invalid date format in command-line argument. Use YYYY-MM-DD.")
                raise

        # Environment variable
        ref_date_env = os.getenv("REFERENCE_DATE")
        if ref_date_env:
            logging.info("Using reference date from environment variable.")
            try:
                return datetime.strptime(ref_date_env, "%Y-%m-%d").date()
            except ValueError:
                logging.error("Invalid date format in environment variable. Use YYYY-MM-DD.")
                raise

        # Dataset
        if df is not None and 'InvoiceDate' in df.columns:
            logging.info("Calculating reference date from dataset.")
            try:
                max_date = df['InvoiceDate'].max()
                return (max_date + pd.DateOffset(1)).date()
            except Exception as e:
                logging.error(f"Failed to calculate reference date from dataset: {e}")
                raise

        # Default to today's date
        if default_to_today:
            logging.warning("Defaulting to today's date as the reference date.")
            return datetime.today().date()

        # Raise an error if no valid date can be determined
        raise ValueError("Unable to determine a valid reference date from available sources.")

    except Exception as e:
        logging.error(f"An error occurred while determining the reference date: {e}")
        raise
# Testing guides
#Command-Line Argument: python file_name.py --reference_data 2024-11-01
# Environment Variable: export REFERENCE_DATE="2024-11-15"
#                       python file_name.py

#==================================================================
# 5. Calculate RFM Metrics
def calculate_rfm(df, reference_date):
    """
    Calculates Recency, Frequency, and Monetary metrics for each customer.

    Args:
        df (pd.DataFrame): Cleaned DataFrame containing transaction data.
        reference_date (datetime.date): Reference date for calculating recency.

    Returns:
        pd.DataFrame: DataFrame with RFM metrics.
    """
    try:
        # Validate required columns
        required_columns = {'CustomerID', 'InvoiceDate', 'InvoiceNo', 'TransactionAmount'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Ensure InvoiceDate is a datetime
        if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
            raise TypeError("Column 'InvoiceDate' must be a datetime type.")

        # Calculate RFM metrics
        rfm = df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (pd.Timestamp(reference_date) - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TransactionAmount', 'sum')
        ).reset_index()

        # Rename columns for clarity (optional, based on your preferences)
        rfm.rename(columns={'CustomerID': 'CustomerID'}, inplace=True)

        logging.info(f"Successfully calculated RFM metrics for {len(rfm)} customers.")
        return rfm

    except Exception as e:
        logging.error(f"Error while calculating RFM metrics: {e}")
        raise


#==================================================================
# 6. Transform and Score RFM Metrics
def log_transform_rfm(rfm):
    """
    Applies log transformation to RFM metrics.
    
    Args:
        rfm (pd.DataFrame): DataFrame containing RFM metrics.

    Returns:
        pd.DataFrame: DataFrame with log-transformed RFM metrics.
    """
    try:
        # Validate required columns
        required_columns = {'Recency', 'Frequency', 'Monetary'}
        missing_columns = required_columns - set(rfm.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Apply log transformation
        for col in required_columns:
            rfm[f'{col}_log'] = np.log1p(rfm[col])

        logging.info("Log transformation applied successfully.")
        return rfm

    except Exception as e:
        logging.error(f"Error during log transformation: {e}")
        raise
#---------------------------------------------------------------
def score_rfm_metrics(rfm, bins=5):
    """
    Assigns scores to log-transformed RFM metrics using quantile-based binning.
    
    Args:
        rfm (pd.DataFrame): DataFrame with log-transformed RFM metrics.
        bins (int): Number of quantile bins for scoring.

    Returns:
        pd.DataFrame: DataFrame with scored RFM metrics.
    """
    try:
        # Validate required columns
        required_columns = {'Recency_log', 'Frequency_log', 'Monetary_log'}
        missing_columns = required_columns - set(rfm.columns)
        if missing_columns:
            raise ValueError(f"Missing required log-transformed columns: {missing_columns}")

        # Assign quantile scores
        rfm['Recency_score'] = pd.qcut(rfm['Recency_log'], bins, labels=range(bins, 0, -1))
        rfm['Frequency_score'] = pd.qcut(rfm['Frequency_log'].rank(method='first'), bins, labels=range(1, bins + 1))
        rfm['Monetary_score'] = pd.qcut(rfm['Monetary_log'].rank(method='first'), bins, labels=range(1, bins + 1))

        # Compute overall RFM score
        rfm['RFM_score'] = rfm[['Recency_score', 'Frequency_score', 'Monetary_score']].sum(axis=1).astype(int)

        logging.info("RFM metrics scored successfully.")
        return rfm

    except Exception as e:
        logging.error(f"Error during RFM scoring: {e}")
        raise


#==================================================================
# 7. Assign Customer Segments
def assign_value_segments(rfm, segment_labels=None):
    """
    Assigns value-based segments using RFM score quantiles.
    
    Args:
        rfm (pd.DataFrame): DataFrame containing RFM scores.
        segment_labels (list, optional): List of segment labels. Defaults to ['Low-Value', 'Mid-Value', 'High-Value'].

    Returns:
        pd.DataFrame: DataFrame with assigned value-based segments.
    """
    try:
        segment_labels = segment_labels or ['Low-Value', 'Mid-Value', 'High-Value']
        num_segments = len(segment_labels)
        
        # Validate inputs
        if 'RFM_score' not in rfm.columns:
            raise ValueError("Column 'RFM_score' not found in the DataFrame.")
        if num_segments < 2:
            raise ValueError("At least two segment labels are required.")

        # Assign segments
        rfm['Value Segment'] = pd.qcut(rfm['RFM_score'], q=num_segments, labels=segment_labels)

        logging.info(f"Value segments assigned successfully with {num_segments} segments.")
        return rfm

    except Exception as e:
        logging.error(f"Error during value segmentation: {e}")
        raise


def define_behavioral_segments(rfm, segment_rules=None):
    """
    Assigns behavioral segments based on RFM score.
    
    Args:
        rfm (pd.DataFrame): DataFrame containing RFM scores.
        segment_rules (list, optional): List of tuples (threshold, label). Defaults to predefined rules.

    Returns:
        pd.DataFrame: DataFrame with assigned behavioral segments.
    """
    try:
        if 'RFM_score' not in rfm.columns:
            raise ValueError("Column 'RFM_score' not found in the DataFrame.")

        # Define default rules if none are provided
        segment_rules = segment_rules or [
            (13, 'Champions'),
            (10, 'Loyal Customers'),
            (7, 'Potential Loyalists'),
            (5, 'At Risk'),
            (0, 'Hibernating'),
        ]

        # Sort rules by threshold descending
        segment_rules.sort(reverse=True, key=lambda x: x[0])

        def classify(row):
            for threshold, label in segment_rules:
                if row['RFM_score'] >= threshold:
                    return label
            return 'Undefined'

        # Apply classification
        rfm['RFM Customer Segment'] = rfm.apply(classify, axis=1)

        # Logging segment distribution
        segment_counts = rfm['RFM Customer Segment'].value_counts()
        logging.info(f"Behavioral segments assigned: {segment_counts.to_dict()}")

        return rfm

    except Exception as e:
        logging.error(f"Error during behavioral segmentation: {e}")
        raise

#==================================================================
# 8. Visualization
# Includes all plotting functions as provided earlier, unchanged.

# ----------------Value Segments Plot----------------------

def plot_value_segment_distribution(rfm, output_path=None, save=False):
    """
    Plots the distribution of customers in each RFM Value Segment using Seaborn.
    
    Args:
        rfm (pd.DataFrame): DataFrame containing the 'Value Segment' column.
        output_path (str, optional): Path to save the plot. Defaults to None.
        save (bool, optional): Whether to save the plot. Defaults to False.
    
    Returns:
        None
    """
    # Count customers in each value segment
    value_segment_count = rfm['Value Segment'].value_counts().reset_index()
    value_segment_count.columns = ['Value Segment', 'Count']
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=value_segment_count,
        x='Value Segment',
        y='Count',
        palette='pastel'
    )
    
    # Add titles and labels
    plt.title('RFM Value Segment Distribution', fontsize=16, weight='bold')
    plt.xlabel('RFM Value Segment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Save the plot if required
    if save and output_path:
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    
    plt.show()

    #-------------------Customer Segments Plot-----------------------------

def plot_behavioral_segment_distribution(rfm, output_path, save=False):
    """
    Generates a bar plot showing the distribution of behavioral segments.

    Parameters:
        rfm (DataFrame): The RFM analysis data.
        output_path (str): Path to save the plot.
        save (bool): Whether to save the plot.
    """
    try:
        segment_counts = rfm['RFM Customer Segment'].value_counts()
        custom_palette = {
            "Champions": "blue",
            "Loyal Customers": "green",
            "Potential Loyalists": "orange",
            "At Risk": "red",
            "Hibernating": "gray",
        }
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=segment_counts.index,
            y=segment_counts.values,
            palette=custom_palette  # Ensure this matches the segment names
        )
        plt.title("Behavioral Segment Distribution")
        plt.ylabel("Number of Customers")
        plt.xlabel("Segment")
        if save:
            plt.savefig(output_path, bbox_inches="tight")
        plt.show()
        logging.info(f"Plot saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate behavioral segment plot: {e}")

#-----------------------Average RFM Scores--------------------------------

def plot_average_rfm_scores(rfm, output_path=None, save=False):
    """
    Plots average Recency, Frequency, and Monetary scores for each RFM customer segment.
    """
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
    
    # Melt the dataframe to long format for Seaborn
    segment_scores_melted = segment_scores.melt(id_vars="RFM Customer Segment", 
                                                value_vars=['Recency_score', 'Frequency_score', 'Monetary_score'], 
                                                var_name='Score Type', 
                                                value_name='Score')
    
    # Set up the corrected color palette
    color_palette = ['#9ecae1', '#5e9ed9', '#206694']  # Updated to valid colors
    
    # Create the grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=segment_scores_melted, 
        x='RFM Customer Segment', 
        y='Score', 
        hue='Score Type', 
        palette=color_palette  # Corrected palette
    )
    
    # Add titles and labels
    plt.title('Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores', fontsize=16, weight='bold')
    plt.xlabel('RFM Segments', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Score Type', title_fontsize='13', loc='upper left')
    
    # Save the plot if required
    if save and output_path:
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    
    plt.show()

#------------------------- Revenue Contribution ---------------------------------

def plot_revenue_contribution(rfm, output_path, save=False):
    """
    Generates a pie chart showing the revenue contribution of customer segments with a legend.

    Parameters:
        rfm (DataFrame): The RFM analysis data.
        output_path (str): Path to save the plot.
        save (bool): Whether to save the plot.
    """
    try:
        revenue_contribution = (
            rfm.groupby("RFM Customer Segment")["Monetary"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        revenue_contribution["Percentage"] = (
            revenue_contribution["Monetary"]
            / revenue_contribution["Monetary"].sum()
            * 100
        )

        custom_palette = {
            "Champions": "blue",
            "Loyal Customers": "green",
            "Potential Loyalists": "orange",
            "At Risk": "red",
            "Hibernating": "gray",
        }
        
        # Pie chart visualization
        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(
            revenue_contribution["Percentage"],
            labels=revenue_contribution["RFM Customer Segment"],
            autopct='%1.1f%%',
            colors=[custom_palette.get(segment, "lightgrey") for segment in revenue_contribution["RFM Customer Segment"]],
            startangle=90,
            wedgeprops={'edgecolor': 'white'}
        )
        
        # Add legend
        plt.legend(
            wedges,
            revenue_contribution["RFM Customer Segment"],
            title="Customer Segments",
            loc="best",
            bbox_to_anchor=(1, 0.5),
            fontsize=10
        )
        
        # Title
        plt.title("Revenue Contribution by Segment")
        
        # Show or save the plot
        if save:
            plt.savefig(output_path, bbox_inches="tight")
        plt.show()
        
        logging.info(f"Plot saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate revenue contribution plot: {e}")


#==================================================================
# 9. Save Results
def save_rfm_results(rfm, output_path):
    """Saves RFM analysis(DataFrame:'rfm') results to a file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rfm.to_pickle(output_path)
        logging.info(f"RFM results saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save RFM results: {e}")

def save_summary_table(data, output_path, mode="w"):
    """Saves summary table(*CUSTOMER REVENUE CONTRIBUTION PROPORTION TABLE*) to a text file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, mode) as file:
            file.write(data)
        logging.info(f"Summary table saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save summary table: {e}")
#==================================================================
# 10. Generate HTML Report
def generate_report(rfm, summary_data, output_html="notebooks/rfm_analysis_report.html"):
    """Generates an HTML report for RFM analysis."""
    try:
        template_path = "report_template.html"
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
        # Open template
        with open(template_path, "r") as file:
            html_template = file.read()

        # Prepare context to template
        template = Template(html_template)
        context = prepare_report_context(rfm, summary_data)
        rendered_html = template.render(context)

        # Write out template and save in 'output_html' directory
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        with open(output_html, "w") as report_file:
            report_file.write(rendered_html)

        logging.info(f"Report generated successfully: {output_html}")

    except FileNotFoundError as e:
        logging.warning(f"Template file not found. Report generation skipped. {e}")
        with open(output_html, "w") as report_file:
            report_file.write("<h1>Report Generation Failed</h1><p>Template not found.</p>")
    except Exception as e:
        logging.error(f"An error occurred while generating the report: {e}")
        
#--------------------Prepare Report Context------------------
# The Function Creates context for your report and returns a dictionary of relevant report content
def prepare_report_context(rfm, summary_data):
    # Overview text
    overview = """
    This report presents the findings of an RFM (Recency, Frequency, Monetary) analysis conducted to evaluate customer behavior and identify actionable insights for business growth.
    Customers were segmented into behavioral and value-based groups, with visualizations highlighting key patterns and opportunities for strategic action.

    The analysis includes revenue contributions, customer distributions across segments, and specific recommendations for engaging and retaining different customer groups.
    """

    # Insights based on RFM analysis
    insights = [
        "Champions contribute the highest revenue at 70.19% despite being 20% of the customer base.",
        "Hibernating customers generate only 1.18% of revenue, highlighting low engagement.",
        "Potential Loyalists and Loyal Customers together contribute 25.71% of the revenue, indicating opportunities for growth.",
        "At Risk customers contribute 2.92% of revenue, warranting strategies to re-engage them."
    ]

    # Recommendations based on insights
    recommendations = [
        "Strengthening relationships with champions...",
        "Nurture loyal customers and potential loyalists...",
        "Re-engage At Risk customers...",
        "Optimize efforts for hibernating customers...",
        "Monitor and track performance..."
    ]
    segment_revenue_dict = summary_data[["RFM Customer Segment", "Revenue Contribution"]].to_dict(orient="records")
    # # Convert summary_data to dictionary for HTML rendering    _____________________(ISSUE NOTED)________________________
    # summary_data[["RFM Customer Segment", "Customer Proportion", "Revenue Contribution"]].to_dict(orient="records")

    # Prepare the context dictionary
    context = {
        "overview": overview,
        "insights": insights,
        "recommendations": recommendations,
        "segment_revenue": segment_revenue_dict,  # Full DataFrame for analysis/plotting
        # Add image paths relative to the report's location
        "plot_revenue": "images/revenue_contribution.png",
        "plot_segments": "images/rfm_segments_comparison.png",
        "plot_value_segments": "images/rfm_value_segment_dist.png",
        "plot_rfm_comparison": "images/rfm_comparisons.png",
    }

    return context



#==================================================================
def main(data_path, output_pkl, summary_txt, report_html="../notebooks/rfm_analysis_report.html"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    try:
        # Load and clean data
        df = load_data(data_path)
        if df is None:
            logging.error("Data loading failed. Exiting workflow.")
            return
        df = clean_data(df)
        
        # Perform RFM analysis
        reference_date = get_reference_date(df)
        logging.info(f"Reference date: {reference_date}")
        rfm = calculate_rfm(df, reference_date)
        rfm = log_transform_rfm(rfm)
        rfm = score_rfm_metrics(rfm)
        rfm = assign_value_segments(rfm)
        rfm = define_behavioral_segments(rfm)
        
        # Save processed RFM data
        os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
        save_rfm_results(rfm, output_pkl)
        logging.info(f"RFM results saved to {output_pkl}")
        
        # Prepare summary data
        summary_data = rfm.groupby("RFM Customer Segment").agg(Customers=("CustomerID", "count"),Revenue=("Monetary", "sum")).reset_index()
        total_customers = summary_data["Customers"].sum()
        total_revenue = summary_data["Revenue"].sum()
        summary_data["Customer Proportion"] = (summary_data["Customers"] / total_customers * 100).round(2).astype(str) + "%"
        summary_data["Revenue Contribution"] = (summary_data["Revenue"] / total_revenue * 100).round(2).astype(str) + "%"
        
        # Generate and save summary table
        summary_table = tabulate(
            summary_data[["RFM Customer Segment", "Customer Proportion", "Revenue Contribution"]].values.tolist(),
            headers=["Segment", "Customer Proportion", "Revenue Contribution"],
            tablefmt="grid"
        )
        save_summary_table(summary_table, summary_txt)
        
        # Generate plots (Plot:Path pairs)
        plot_paths = {
            "value_segment": "images/rfm_value_segment_dist.png",
            "behavioral_segment": "images/rfm_segments_comparison.png",
            "rfm_comparisons": "images/rfm_comparisons.png",
            "revenue_contribution": "images/revenue_contribution.png",
        }
        for plot, path in plot_paths.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
        logging.info("Generating plots...")
        plot_value_segment_distribution(rfm, plot_paths["value_segment"], save=True)
        plot_behavioral_segment_distribution(rfm, plot_paths["behavioral_segment"], save=True)
        plot_average_rfm_scores(rfm, plot_paths["rfm_comparisons"], save=True)
        plot_revenue_contribution(rfm, plot_paths["revenue_contribution"], save=True)
        logging.info("Plots saved successfully.")
        
        # Generate the report
        context = prepare_report_context(rfm, summary_data)
        generate_report(rfm, summary_data, output_html=report_html)
        logging.info(f"Report generated: {os.path.relpath(report_html)}")
        
    except Exception as e:
        logging.error(f"An error occurred during the workflow: {e}")

if __name__ == "__main__":
    CONFIG = {
        "data_path": "data/raw/online_retail.csv",
        "output_pkl": "data/processed/rfm_table.pkl",
        "summary_txt": "images/segment_comparison_table.txt",
        "report_html": "../notebooks/rfm_analysis_report.html",
    }
    main(
        data_path=CONFIG["data_path"],
        output_pkl=CONFIG["output_pkl"],
        summary_txt=CONFIG["summary_txt"],
        report_html=CONFIG["report_html"]
    )


