import pandas as pd

# Load your dataset
df = pd.read_csv('./data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
# Columns you want to display

# remove leading/trailing spaces
df.columns = df.columns.str.strip()

print(df.columns.tolist())
cols = [
    "Bwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bulk Rate",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd URG Flags"
]

# Display only these columns
print(df[cols])