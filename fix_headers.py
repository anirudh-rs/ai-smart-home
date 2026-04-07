import pandas as pd

df = pd.read_csv('data/events.csv')
print("Current columns:", df.columns.tolist())
print("First row:", df.iloc[0].tolist())