"""
Creates data/loan_approval_sample.csv  (1,200 rows, Kaggle "Loan Approval" style schema).

This is a STAND-IN so the project runs offline out of the box. For your submission,
download a real dataset from Kaggle (e.g. "Loan Prediction Problem Dataset") and upload
that file in the web app instead - the pipeline works on any CSV/XLSX.

The file deliberately contains missing values, duplicate rows and outliers so every
preprocessing step has something real to do.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1200

gender = rng.choice(["Male", "Female"], n, p=[0.8, 0.2])
married = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])
dependents = rng.choice(["0", "1", "2", "3+"], n, p=[0.57, 0.17, 0.16, 0.10])
education = rng.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22])
self_emp = rng.choice(["Yes", "No"], n, p=[0.14, 0.86])
income = np.round(rng.lognormal(8.5, 0.55, n)).astype(float)
co_income = np.where(rng.random(n) < 0.4, 0, np.round(rng.lognormal(7.6, 0.7, n)))
loan_amt = np.round(income / 40 + co_income / 80 + rng.normal(90, 35, n)).clip(9, None)
term = rng.choice([12, 36, 60, 84, 120, 180, 240, 300, 360, 480], n,
                  p=[.01, .01, .01, .01, .02, .07, .02, .01, .82, .02]).astype(float)
credit = rng.choice([1.0, 0.0], n, p=[0.84, 0.16])
area = rng.choice(["Urban", "Semiurban", "Rural"], n, p=[0.33, 0.38, 0.29])

score = (3.0 * credit
         + 0.9 * (area == "Semiurban") + 0.2 * (area == "Urban")
         + 0.5 * (education == "Graduate") + 0.3 * (married == "Yes")
         + 0.00012 * (income + co_income)
         - 0.008 * loan_amt
         - 0.1 * (self_emp == "Yes")
         - 2.2 + rng.normal(0, 0.7, n))
status = np.where(score > 0.9, "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{100000 + i}" for i in range(n)],
    "Gender": gender, "Married": married, "Dependents": dependents,
    "Education": education, "Self_Employed": self_emp,
    "ApplicantIncome": income, "CoapplicantIncome": co_income,
    "LoanAmount": loan_amt, "Loan_Amount_Term": term,
    "Credit_History": credit, "Property_Area": area, "Loan_Status": status,
})

# --- inject realistic dirt -------------------------------------------------
for col, frac in [("Gender", .02), ("Married", .01), ("Dependents", .025),
                  ("Self_Employed", .05), ("LoanAmount", .035),
                  ("Loan_Amount_Term", .02), ("Credit_History", .08)]:
    df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
idx = rng.choice(n, 12, replace=False)
df.loc[idx, "ApplicantIncome"] = df.loc[idx, "ApplicantIncome"] * rng.integers(6, 14, 12)   # outliers
df.loc[rng.choice(n, 8, replace=False), "LoanAmount"] *= 5
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)                     # duplicates
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/loan_approval_sample.csv", index=False)
print("saved data/loan_approval_sample.csv", df.shape)
