# **RFM Analysis Project**

## **Overview**

This project focuses on implementing **RFM (Recency, Frequency, Monetary) analysis** to segment customers for small and medium retailers. The goal is to help retailers gain actionable insights into their customer base and improve marketing and retention strategies.

---

## **Objectives**

1. Calculate RFM metrics for each customer:
   - **Recency**: Time since the last transaction.
   - **Frequency**: Total number of transactions.
   - **Monetary**: Total spending.
2. Score customers based on RFM values.
3. Segment customers into actionable groups (e.g., VIPs, At-Risk).
4. Visualize customer distributions and segment insights.
5. Export results for further use.

---

## **Data Description**

The dataset includes the following key features:

- **Customer ID**: Unique identifier for each customer.
- **Transaction Date**: Date of each transaction.
- **Transaction Amount**: Revenue generated from the transaction.

---

## **Project Workflow**

### **Step 1: Data Loading and Exploration**

- Load the dataset using Python’s `pandas`.
- Inspect for missing or duplicate values.
- Convert `Transaction Date` to datetime format.

### **Step 2: Data Preprocessing**

- Remove missing/invalid records.
- Aggregate transactions by `Customer ID` to compute RFM metrics.

### **Step 3: RFM Metrics Calculation**

- **Recency**: Days since the most recent transaction.
- **Frequency**: Count of transactions.
- **Monetary**: Sum of transaction amounts.

### **Step 4: Scoring and Segmentation**

- Assign RFM scores (1–5) to each metric based on percentile ranking.
- Combine scores to create RFM segments (e.g., R5F5M5 for top customers).
- Label customer tiers (e.g., VIPs, At-Risk).

### **Step 5: Visualization**

- Create distribution plots for Recency, Frequency, and Monetary.
- Visualize customer segments with bar charts or pie charts.

### **Step 6: Results Export**

- Save the RFM table with segments to a CSV file for further use.

---

## **Tech Stack**

- **Programming Language**: Python
- **Libraries**:
  - `pandas`: Data manipulation.
  - `numpy`: Numerical computations.
  - `matplotlib` and `seaborn`: Data visualization.
  - `datetime`: Handling date formats.

---

## **Files and Directory Structure**

```grapql
RFM-Analysis/
│
├── data/
│   └── retail_data.csv                # Input dataset
│
├── src/
│   ├── rfm_analysis.py                # Main RFM analysis script
│   ├── data_preprocessing.py          # Data cleaning and preprocessing script
│   └── visualization.py               # Visualization functions
│
├── output/
│   ├── rfm_analysis_results.csv       # Final RFM table with segments
│   └── plots/                         # Generated plots
│
├── README.md                          # Project overview and instructions
└── requirements.txt                   # List of required libraries

---

## **How to Run**

1. Clone the repository or download the project files.
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
3. Place the dataset in the data/ directory.

4. Run the analysis script:

   ```bash
   python src/rfm_analysis.py
5. Check the output directory for results and visualizations.

---

## **Next Steps**

- Integrate customer recommendations (e.g., marketing strategies for each segment).
- Build a user-friendly interface for uploading data and viewing results.
- Expand the tool to include predictive analytics and inventory insights

## Contact

If you have quastions or need support with this project, feel free to reach out
