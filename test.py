# """
# Summarize CIC-IDS2017 preprocessed day-part CSVs and write a JSON report.
# """

# from __future__ import annotations

# import json
# from collections import Counter
# from datetime import datetime
# from pathlib import Path
# import sys

# import pandas as pd

# BASE = Path(__file__).parent
# OUT_DIR = BASE / "data_preprocessed"

# # Core CIC-IDS2017 per-day files used by the Streamlit dashboard
# FILES = {
#     "Mon+Tue": OUT_DIR / "MonTue-WorkingHours.pcap_ISCX.csv",
#     "Wednesday": OUT_DIR / "Wednesday-workingHours.pcap_ISCX.csv",
#     "Thursday": OUT_DIR / "Thursday-WorkingHours-Combined.pcap_ISCX.csv",
#     "Friday": OUT_DIR / "Friday-WorkingHours-Combined.pcap_ISCX.csv",
# }

# REPORTS_DIR = BASE / "new_reports"


# def _clean_labels(series: pd.Series) -> pd.Series:
#     """Normalize label strings so BENIGN is consistent and stray whitespace is removed."""
#     return series.astype(str).str.strip().str.replace("\ufffd", "-", regex=False)


# def _generate_report(output_name: str) -> dict:
#     """
#     Read the four consolidated CSVs, compute per-file and overall class distributions,
#     and write a JSON report to new_reports/<output_name>.
#     """
#     REPORTS_DIR.mkdir(parents=True, exist_ok=True)

#     file_reports: dict[str, dict] = {}
#     overall_counts = Counter()
#     overall_rows = 0
#     columns_union: set[str] = set()
#     columns_intersection: set[str] | None = None

#     for name, path in FILES.items():
#         df = pd.read_csv(path)
#         df["Label"] = _clean_labels(df["Label"])

#         label_counts = Counter(df["Label"].value_counts(dropna=False).to_dict())
#         benign = int(label_counts.get("BENIGN", 0))
#         attacks = int(len(df) - benign)
#         attack_types = {k: int(v) for k, v in label_counts.items() if k != "BENIGN"}

#         file_reports[name] = {
#             "path": str(path),
#             "rows": int(len(df)),
#             "cols": int(df.shape[1]),
#             "benign": benign,
#             "attacks": attacks,
#             "attack_types": attack_types,
#             "label_counts": {k: int(v) for k, v in label_counts.items()},
#         }

#         overall_counts.update(label_counts)
#         overall_rows += len(df)

#         cols = set(df.columns)
#         columns_union |= cols
#         columns_intersection = cols if columns_intersection is None else columns_intersection & cols

#     benign_total = int(overall_counts.get("BENIGN", 0))
#     attacks_total = int(overall_rows - benign_total)
#     attack_types_total = {k: int(v) for k, v in overall_counts.items() if k != "BENIGN"}

#     report = {
#         "generated_at": datetime.utcnow().isoformat() + "Z",
#         "files": file_reports,
#         "overall": {
#             "total_rows": int(overall_rows),
#             "benign": benign_total,
#             "attacks": attacks_total,
#             "attack_types": attack_types_total,
#             "label_counts": {k: int(v) for k, v in overall_counts.items()},
#             "columns_union_count": len(columns_union),
#             "columns_intersection_count": len(columns_intersection or set()),
#             "columns_match": len(columns_union) == len(columns_intersection or set()),
#         },
#         "source_files": {k: str(v) for k, v in FILES.items()},
#     }

#     # Ensure .json extension and absolute target inside new_reports
#     out_name = output_name if output_name.endswith(".json") else f"{output_name}.json"
#     out_path = REPORTS_DIR / out_name
#     with out_path.open("w", encoding="utf-8") as f:
#         json.dump(report, f, indent=2)

#     print(f"Saved summary to {out_path}")
#     return report


# def datapreprocessed_report():
#     """Generate the report requested for preprocessed files: new_reports/datapreprocessed_rport.json."""
#     _generate_report("datapreprocessed_rport")


# def main():
#     # Default: run the data-preprocessed report
#     datapreprocessed_report()


# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         arg = sys.argv[1].lower()
#         if arg.startswith("data"):
#             datapreprocessed_report()
#         else:
#             datapreprocessed_report()
#     else:
#         main()




# import json
# import ast
# import re
# import os
# from pathlib import Path

# def clean_and_restore_json(directory_path):
#     # Target the specific files that were causing the errors
#     target_files = [
#         "All_finetune_history_raw.json",
#         "All_pretrain_history_raw.json"]
    
#     dir_path = Path(directory_path)
    
#     if not dir_path.exists():
#         print(f"Error: Directory '{directory_path}' not found.")
#         return

#     for filename in target_files:
#         file_path = dir_path / filename
        
#         if not file_path.exists():
#             print(f"Skipping: {filename} (File not found)")
#             continue

#         print(f"Cleaning {filename}...")

#         try:
#             # 1. Read the broken "Python-string" content
#             with open(file_path, 'r') as f:
#                 content = f.read()

#             # 2. Use Regex to strip 'np.float64(...)' and 'np.int64(...)'
#             # This turns 'np.float64(0.025)' into '0.025'
#             clean_content = re.sub(r'np\.\w+\((.*?)\)', r'\1', content)

#             # 3. Convert Python string representation to an actual dictionary
#             # literal_eval is safer than eval()
#             data = ast.literal_eval(clean_content)

#             # 4. Write it back as a pretty-printed, valid JSON file
#             with open(file_path, 'w') as f:
#                 json.dump(data, f, indent=4)

#             print(f"✅ Successfully restored: {filename}")

#         except Exception as e:
#             print(f"❌ Failed to process {filename}: {e}")

# if __name__ == "__main__":
#     # Point this to your 'Save_history' folder relative to this script
#     # Based on your image, it is in the same directory or a subfolder
#     target_folder = "./Save_history" 
#     clean_and_restore_json(target_folder)


from pathlib import Path
import pandas as pd
import numpy as np
FOLDER = Path("/home/waleed64/Documents/Tabnet-ssl/tabnet/filter-data")

total_rows_all = 0
total_nan_rows_all = 0

column_list= ['col1','col2','col3']
list_data= list(np.random.randint(0,10,(100,3)))
data_list= np.array(list_data)
Dataframe1= pd.DataFrame(data=data_list, columns=column_list)
print(f"2D dataframe having three columns:\n{Dataframe1.head(5)}")
total_count= len(Dataframe1)
nan_mask= Dataframe1.isna().any(axis=0)
print(f"nan mask rows:{nan_mask} and length:{total_count}")
summation_nun_max= int(np.sum(nan_mask))

Dataframe1.iloc[0:5,0:2]= np.nan

Dataframe1.loc[10:15,['col2','col3']]
Dataframe1.loc[10:15,['col1','col2','col3']]= 0
print(f"first few rows of dataframe:{Dataframe1.loc[10:15,['col1','col2','col3']]}")
#lets sums null values for each columns..
nan_mask_per_column = Dataframe1.isna().sum()
duplicated_rows = Dataframe1.duplicated().sum()
print(f"duplciated rows:{duplicated_rows}")
duplicates = Dataframe1[Dataframe1.duplicated()].columns
print(f"duplicateds:{duplicates}")
print(f"return mask per column:{nan_mask_per_column}")

cols= ['age','salary','grade','married']
categorical= ['A','B','C','D','E']
binary = [True, False]
size = 1000

data = {'age':np.random.randint(25,60, size), 
        'salary': np.random.randint(40000,90000, size),
          'grade':np.random.choice(categorical,size=size), 
          'married': np.random.choice(binary,size=size)
        }

df =  pd.DataFrame(data=data)

print(f"Here is the latest data:{df.head(5)}")
df.iloc[100:105,0:4]= np.nan
df.iloc[200:205,0:5]= 5

print(f"types of columns:{df.dtypes}")
#seperate numercolumns from cateogircal columns ..
categorical_columns= df.select_dtypes(include=['object']).columns.to_list()
numerical_columns = df.select_dtypes(include=['float64','int64','int32','string']).columns.to_list()
print(f"categorical columns include:{categorical_columns}")
print(f"numerical columns include:{numerical_columns}")

count_unique_category = df['grade'].unique().tolist()

print(f"grade columns with unique categoiry:{count_unique_category} and its count:{len(count_unique_category)}")

df['conditional_col']= np.where(df['salary']>50000, 'high', 'low')
# Using a lambda on the specific column
df['conditional_col'] = df['salary'].apply(lambda x: 'high' if x >= 50000 else 'low')
# Initialize the column with a default value
df['conditional_col'] = 'low'

# Update only the rows that meet the criteria
df.loc[df['salary'] >= 50000, 'conditional_col'] = 'high'

print(f"number of columns:{df.columns} and df:\n{df.head(10)}")



# pracitice isinstance() , here you will placed class and then its object
# and isattriute(), here you will placed object and then its attribute
# also come to know about getattribute()

import json 
from pathlib import Path
import csv 
from typing import Dict, Tuple, List

class animal:
    def __init__(self,name: str, species: str, age:int, color: str, weight: float, breed: str, tasks_list: list, animal_functionality: dict):
        self.name= name
        self.species= species
        self.age= age
        self.color= color
        self.weight= weight
        self.breed= breed
        self.tasks_list= tasks_list
        self.animal_functionality= animal_functionality

        if isinstance(self.color, int):
            print(f"yes the self.color is an instance of int:{self.age}")

        if isinstance(self.age, int):
            print(f"yes the self.age is an instance of int:{self.age}")
        
        if isinstance(self.tasks_list, list):
            print(f"yes the list of animals is an instance of list:{self.age}")

        if isinstance(self.animal_functionality, dict):
            print(f"yes the animal functionality is an instance of dict:{self.age}")
    
    def reproduction(self):
        pass

    def feeding(self):
        pass

    def movement(self):
        pass

class goat(animal):
    def __init__(self, sound: str, has_horns: bool, milk_production: bool):
        self.sound= sound
        self.has_horns= has_horns
        self.milk_production= milk_production
        if isinstance(self.milk_production, bool):
            print(f"self.milk_production is a bool type:{self.milk_production}")
    
    def diary_meet_fiber(self):
        if isinstance(self.sound, str):
            print(f"sound is the instance of the string:{self.sound}")
    
    def climbing_and_grazzing(self, object: object):
        self.object = object
        if isinstance(self.object, goat) or isinstance(self.object, animal):
            print(f"self.object is the real object of goat")


            
if __name__ == "__main__":

    object1 = animal("horse", "lion", 5, 'red', 23.4, 'german sheperad', ['run','make_sound','sit','fight'], {'lion':'fight','dog':'loyality','horse':'running'})

    object2 = goat('maaaa',True, False)
    object2.diary_meet_fiber()
    object2.climbing_and_grazzing(object1)

    if hasattr(object1,'name'):
        print("name")
    
    if hasattr(object1,'reproduction'):
        print(f"reproduction")

    
    if hasattr(object1,'feeding'):
        print("feeding:")

    
    if hasattr(object1,'age'):
        print("age")

    
    if hasattr(object2,'sound'):
        print("sound:")
    
    if hasattr(object2,'climbing_and_grazzing'):
        print("climbing and grazzing:")

# now move into the grouping mechanism...

# axis=0 or axis="index" or axis=1 or axis="columns"
# for csv_path in sorted(FOLDER.glob("*.csv")):

#     print("=" * 100)
#     print(f"File: {csv_path.name}")

#     df = pd.read_csv(csv_path, low_memory=False)

#     total_rows = len(df)
#     nan_mask = df.isna().any(axis=1)
#     nan_rows = int(nan_mask.sum())

#     null_per_column = df.isna().sum()
#     null_per_column = null_per_column[null_per_column > 0].sort_values(ascending=False)

#     print(f"Total rows            : {total_rows:,}")
#     print(f"Rows with any null    : {nan_rows:,}")
#     print(f"Rows without null     : {total_rows - nan_rows:,}")

#     if not null_per_column.empty:
#         print("\nTop columns with maximum null values:")
#         print(null_per_column.head(20).to_string())
#     else:
#         print("\nNo null values in this file.")

#     total_rows_all += total_rows
#     total_nan_rows_all += nan_rows

# print("=" * 100)
# print("OVERALL SUMMARY")
# print(f"Total rows across files         : {total_rows_all:,}")
# print(f"Total rows with any null        : {total_nan_rows_all:,}")
# print(f"Total rows without any null     : {total_rows_all - total_nan_rows_all:,}")
