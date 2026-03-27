# ShaadiSpend Analytics — Streamlit Dashboard

## Setup & Run Instructions

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the dashboard
```bash
streamlit run app.py
```

The dashboard will open automatically at: http://localhost:8501

---

## Folder Structure
```
shaadispend_dashboard/
├── app.py                          # Main Streamlit app (entry point)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── ShaadiSpend_RAW_2000.csv    # Raw dataset (with dirty data)
│   └── ShaadiSpend_CLEANED_2000.csv # Cleaned + feature-engineered dataset
└── pages/
    ├── 1_Overview.py               # Market overview & KPI cards
    ├── 2_EDA.py                    # Descriptive analytics & 12 charts
    ├── 3_Correlation.py            # Pearson correlation heatmap
    ├── 4_Clustering.py             # K-Means customer persona discovery
    ├── 5_Classification.py         # Random Forest — predict platform intent
    ├── 6_Regression.py             # Linear/Ridge regression — predict budget
    └── 7_ARM.py                    # Association Rule Mining — vendor baskets
```

## Dashboard Sections

| Page | Analytics | Business Question |
|---|---|---|
| Overview | Descriptive | What does the market look like? |
| EDA | Descriptive | What patterns exist in the data? |
| Correlation | Descriptive | Which variables are related? |
| Clustering | Unsupervised | Who are our customer personas? |
| Classification | Supervised | Will this customer use ShaadiSpend? |
| Regression | Supervised | What budget will this customer have? |
| ARM | Unsupervised | Which vendor services are bought together? |
