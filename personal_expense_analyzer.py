# ============================================================
# PERSONAL EXPENSE ANALYZER
# Python for Data Science - GTU PBL Micro Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 1. PROJECT TITLE
# ============================================================

print("=" * 70)
print("              PERSONAL EXPENSE ANALYZER")
print("              Python for Data Science - GTU PBL")
print("=" * 70)


# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "expenses.csv"

try:
    df = pd.read_csv(file_path)
    print("\nDataset loaded successfully!")

except FileNotFoundError:
    print("\nERROR: expenses.csv not found.")
    print("Please keep expenses.csv in the same folder as this Python file.")
    exit()


# ============================================================
# 3. DISPLAY FIRST 5 RECORDS
# ============================================================

print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# 4. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nNumber of Transactions:", len(df))

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 5. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)


# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


# Check duplicate records
print("\nDuplicate Records:",
      df.duplicated().sum())


# Remove duplicate records
df = df.drop_duplicates()


# Convert Date column into proper date format
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)


# Convert Amount into numeric
df["Amount"] = pd.to_numeric(
    df["Amount"],
    errors="coerce"
)


# Fill missing amounts using median
df["Amount"] = df["Amount"].fillna(
    df["Amount"].median()
)


# Remove rows with invalid dates
df = df.dropna(
    subset=["Date"]
)


# Clean text columns
df["Category"] = df["Category"].astype(str).str.strip()
df["Description"] = df["Description"].astype(str).str.strip()


# Remove invalid negative expenses
df = df[df["Amount"] >= 0]


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


print("\nData cleaning completed successfully!")


# ============================================================
# 6. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("BASIC EXPENSE STATISTICS")
print("=" * 70)


total_expense = df["Amount"].sum()

average_transaction = df["Amount"].mean()

highest_expense = df["Amount"].max()

lowest_expense = df["Amount"].min()

number_of_transactions = len(df)

number_of_days = df["Date"].nunique()


print(
    f"\nTotal Expense        : ₹{total_expense:.2f}"
)

print(
    f"Average Transaction  : ₹{average_transaction:.2f}"
)

print(
    f"Highest Expense      : ₹{highest_expense:.2f}"
)

print(
    f"Lowest Expense       : ₹{lowest_expense:.2f}"
)

print(
    f"Total Transactions   : {number_of_transactions}"
)

print(
    f"Days with Expenses   : {number_of_days}"
)


# ============================================================
# 7. AVERAGE DAILY EXPENDITURE
# ============================================================

daily_expense = df.groupby(
    "Date"
)["Amount"].sum()


average_daily_expense = daily_expense.mean()


print(
    f"\nAverage Daily Expense: ₹{average_daily_expense:.2f}"
)


# ============================================================
# 8. CATEGORY-WISE ANALYSIS
# ============================================================

category_expense = (
    df.groupby("Category")["Amount"]
    .sum()
    .sort_values(ascending=False)
)


print("\n" + "=" * 70)
print("CATEGORY-WISE EXPENSE")
print("=" * 70)


print(
    category_expense
)


# ============================================================
# 9. HIGHEST SPENDING CATEGORY
# ============================================================

highest_category = category_expense.idxmax()

highest_category_amount = category_expense.max()


print(
    f"\nHighest Spending Category : "
    f"{highest_category}"
)

print(
    f"Amount Spent              : "
    f"₹{highest_category_amount:.2f}"
)


# ============================================================
# 10. HIGHEST SPENDING DAY
# ============================================================

highest_spending_day = daily_expense.idxmax()

highest_spending_day_amount = daily_expense.max()


print(
    f"\nHighest Spending Day      : "
    f"{highest_spending_day.strftime('%d-%m-%Y')}"
)

print(
    f"Amount Spent on that Day  : "
    f"₹{highest_spending_day_amount:.2f}"
)


# ============================================================
# 11. TOP 5 EXPENSES
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 EXPENSES")
print("=" * 70)


top_5_expenses = df.sort_values(
    "Amount",
    ascending=False
).head(5)


print(
    top_5_expenses[
        [
            "Date",
            "Category",
            "Description",
            "Amount"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. CATEGORY-WISE PERCENTAGE
# ============================================================

category_percentage = (
    category_expense / total_expense
) * 100


category_percentage = category_percentage.round(2)


print("\n" + "=" * 70)
print("CATEGORY-WISE EXPENSE PERCENTAGE")
print("=" * 70)


print(category_percentage)


# ============================================================
# 13. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)


print(
    df["Amount"].describe()
)


# ============================================================
# 14. MOST EXPENSIVE TRANSACTION
# ============================================================

most_expensive_transaction = df.loc[
    df["Amount"].idxmax()
]


print("\n" + "=" * 70)
print("MOST EXPENSIVE TRANSACTION")
print("=" * 70)


print(
    f"\nDate        : "
    f"{most_expensive_transaction['Date'].strftime('%d-%m-%Y')}"
)

print(
    f"Category    : "
    f"{most_expensive_transaction['Category']}"
)

print(
    f"Description : "
    f"{most_expensive_transaction['Description']}"
)

print(
    f"Amount      : "
    f"₹{most_expensive_transaction['Amount']:.2f}"
)


# ============================================================
# 15. DAILY EXPENSE TABLE
# ============================================================

print("\n" + "=" * 70)
print("DAILY EXPENSE SUMMARY")
print("=" * 70)


daily_summary = daily_expense.reset_index()

daily_summary.columns = [
    "Date",
    "Total_Daily_Expense"
]


print(
    daily_summary.to_string(index=False)
)


# ============================================================
# 16. SAVE ANALYZED DATA
# ============================================================

output_folder = "output"

# Create output folder if it does not exist
os.makedirs(
    output_folder,
    exist_ok=True
)


# Create a copy for saving
analyzed_df = df.copy()


# Convert date back to readable format
analyzed_df["Date"] = analyzed_df[
    "Date"
].dt.strftime("%d-%m-%Y")


# Add category percentage
analyzed_df["Category_Percentage"] = (
    analyzed_df["Category"]
    .map(category_percentage)
)


# Save analyzed dataset
output_csv = (
    "output/analyzed_expenses.csv"
)


analyzed_df.to_csv(
    output_csv,
    index=False
)


print(
    f"\nAnalyzed dataset saved to: "
    f"{output_csv}"
)


# ============================================================
# 17. VISUALIZATION
# ============================================================

sns.set_theme(
    style="whitegrid"
)


# Create one dashboard containing 3 graphs
fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 11)
)


# ============================================================
# GRAPH 1 - PIE CHART
# EXPENSE BY CATEGORY
# ============================================================

category_expense.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    ax=axes[0, 0]
)


axes[0, 0].set_title(
    "Expense Distribution by Category",
    fontsize=14,
    fontweight="bold"
)


axes[0, 0].set_ylabel("")


# ============================================================
# GRAPH 2 - BAR CHART
# CATEGORY-WISE EXPENSE
# ============================================================

sns.barplot(
    x=category_expense.values,
    y=category_expense.index,
    ax=axes[0, 1]
)


axes[0, 1].set_title(
    "Category-wise Expenditure",
    fontsize=14,
    fontweight="bold"
)


axes[0, 1].set_xlabel(
    "Amount (₹)"
)


axes[0, 1].set_ylabel(
    "Category"
)


# ============================================================
# GRAPH 3 - LINE CHART
# EXPENSE OVER TIME
# ============================================================

axes[1, 0].plot(
    daily_expense.index,
    daily_expense.values,
    marker="o"
)


axes[1, 0].set_title(
    "Daily Expenditure Over Time",
    fontsize=14,
    fontweight="bold"
)


axes[1, 0].set_xlabel(
    "Date"
)


axes[1, 0].set_ylabel(
    "Daily Expense (₹)"
)


axes[1, 0].tick_params(
    axis="x",
    rotation=45
)


# ============================================================
# GRAPH 4 - TOP 5 EXPENSES
# ============================================================

top_5_plot = top_5_expenses.copy()


top_5_plot["Label"] = (
    top_5_plot["Category"]
    + " - "
    + top_5_plot["Description"]
)


sns.barplot(
    data=top_5_plot,
    x="Amount",
    y="Label",
    ax=axes[1, 1]
)


axes[1, 1].set_title(
    "Top 5 Individual Expenses",
    fontsize=14,
    fontweight="bold"
)


axes[1, 1].set_xlabel(
    "Amount (₹)"
)


axes[1, 1].set_ylabel(
    "Expense"
)


# ============================================================
# 18. MAIN TITLE
# ============================================================

plt.suptitle(
    "Personal Expense Analysis Dashboard",
    fontsize=20,
    fontweight="bold"
)


# ============================================================
# 19. ADJUST LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)


# ============================================================
# 20. SAVE DASHBOARD
# ============================================================

dashboard_path = (
    "output/expense_analysis_dashboard.png"
)


plt.savefig(
    dashboard_path,
    dpi=300,
    bbox_inches="tight"
)


print(
    f"Dashboard saved to: "
    f"{dashboard_path}"
)


# ============================================================
# 21. SHOW DASHBOARD
# ============================================================

plt.show()


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    f"\nTotal Monthly Expenditure : ₹{total_expense:.2f}"
)

print(
    f"Average Daily Expenditure : ₹{average_daily_expense:.2f}"
)

print(
    f"Highest Spending Category : {highest_category}"
)

print(
    f"Highest Spending Day      : "
    f"{highest_spending_day.strftime('%d-%m-%Y')}"
)

print(
    f"Highest Single Expense    : ₹{highest_expense:.2f}"
)

print("\nOutput files:")
print("1. output/analyzed_expenses.csv")
print("2. output/expense_analysis_dashboard.png")

print("\nThank you!")