import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import subprocess

print("\n--- Step 1: Creating Movie Dataset ---")
data = {
    "budget": [50, 30, 80, 10, 60, 120, 40],
    "marketing": [20, 15, 40, 5, 30, 50, 10],
    "revenue": [150, 80, 200, 25, 160, 300, 90]
}
df = pd.DataFrame(data)
print(df)

print("\n--- Step 2: Load Data into SQLite (SQL Skill) ---")
conn = sqlite3.connect(":memory:")
df.to_sql("movies", conn, index=False, if_exists="replace")

query = "SELECT budget, marketing, revenue FROM movies WHERE budget > 20;"
print("\nSQL Query:", query)

sql_df = pd.read_sql_query(query, conn)
print("\nFiltered SQL Data:\n", sql_df)

print("\n--- Step 3: Train ML Model (scikit-learn) ---")
X = sql_df[["budget", "marketing"]]
y = sql_df["revenue"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict(np.array([[70, 25]]))[0]
print(f"\nPredicted revenue for budget=70, marketing=25: {prediction:.2f}")

print("\n--- Step 4: Plot Data (Matplotlib) ---")
plt.scatter(sql_df["budget"], sql_df["revenue"])
plt.title("Budget vs Revenue")
plt.xlabel("Budget (M)")
plt.ylabel("Revenue (M)")
plt.grid(True)
plt.show()

print("\n--- Step 5: Run a Bash Command (Linux + Bash Skill) ---")
result = subprocess.run("echo Pipeline executed successfully!", shell=True, capture_output=True, text=True)
print("Bash output:", result.stdout)

print("\n--- Step 6: Dockerfile (printed only) ---")
dockerfile = """
FROM python:3.10
WORKDIR /app
COPY movie_pipeline.py .
RUN pip install pandas numpy scikit-learn matplotlib
CMD ["python", "movie_pipeline.py"]
"""
print(dockerfile)

print("\n--- Step 7: Cloud Command Simulation (AWS/GCP/Azure) ---")
print("Example AWS upload: aws s3 cp movie_pipeline.py s3://mybucket/")

print("\nDone! Entire pipeline ran inside ONE file.")
