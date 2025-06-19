import numpy as np
import pandas as pd

input_file = 'dat_md.csv'
output_file = 'dat_md-mean.csv'
df = pd.read_csv(input_file)

mapping_rules = {
    'DestinationDischarge': {
        'Home': 3,
        'HealthcareFacility': 2,
        'Unknown': 1,
        'Died': 0
    },
    'admission.ward': {
        'Cardiology': 1,
        'GeneralWard': 0,
        'ICU': 1,
        'Others': 1,
    },
    'admission.way': {
        'NonEmergency': 1,
        'Emergency': 0
    },
    'occupation': {
        'UrbanResident': 5,
        'Officer': 4,
        'worker': 3,
        'farmer': 2,
        'Others': 1,
        'NA': 0
    },
    'discharge.department': {
        'Cardiology': 3,
        'GeneralWard': 2,
        'ICU': 1,
        'Others': 0
    },
    'gender': {
        'Male': 0,
        'Female': 1
    },
    'type.of.heart.failure': {
        'Both': 2,
        'Left': 1,
        'Right': 0
    },
    'NYHA.cardiac.function.classification': {
        'IV': 4,
        'III': 3,
        'II': 2
    },
    'Killip.grade': {
        'IV': 4,
        'III': 3,
        'II': 2,
        'I': 1
    },
    'type.II.respiratory.failure': {
        'NonTypeII': 0,
        'TypeII': 1
    },
    'consciousness': {
        'Clear': 3,
        'ResponsiveToSound': 2,
        'ResponsiveToPain': 1,
        'Nonresponsive': 0
    },
    'respiratory.support.': {
        'NIMV': 2,
        'IMV': 1,
        'None': 0
    },
    'oxygen.inhalation': {
        'AmbientAir': 1,
        'OxygenTherapy': 0
    }
}

# Application replacement rules
unmapped_columns = []
unmapped_values = {}
for column, mapping in mapping_rules.items():
    if column in df.columns:
        pre_values = df[column].unique()
        df[column] = df[column].map(mapping)
        post_values = df[column].unique()
        unmapped = [val for val in pre_values if val not in mapping and pd.notna(val)]

        if unmapped:
            unmapped_values[column] = unmapped
            print(f"warning: Column '{column}' exist unmapped columns: {unmapped}")
            df.loc[~df[column].notna() & df[column].notnull(), column] = float('nan')
    else:
        unmapped_columns.append(column)
        print(f"warning: do not exist column: '{column}'，Skip processing")
if 'occupation' in df.columns:
    df['occupation'] = pd.to_numeric(df['occupation'], errors='coerce').fillna(0)
# Filter the columns with null values exceeding 20% in the feature column
print("\n" + "=" * 50)
print("Filter the columns with null values exceeding 20% in the feature column")
print("=" * 50)

total_columns = len(df.columns)
if total_columns < 154:
    cols_to_check = df.columns.tolist()
else:
    cols_to_check = df.columns[:154].tolist()

columns_to_drop = []
for col in cols_to_check:
    total_cells = len(df)
    null_count = df[col].isnull().sum()
    null_percentage = (null_count / total_cells) * 100

    if null_percentage > 20:
        columns_to_drop.append(col)
        print(f"Delete column: '{col}': Null value ratio {null_percentage:.2f}% > 20%")

if columns_to_drop:
    df.drop(columns=columns_to_drop, inplace=True)
else:
    print("No columns were deleted")
# Fill in the mean values of the remaining columns (all numeric columns)
print("\n" + "=" * 50)
print("=" * 50)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    print("Warning: The numeric column was not found. Skip the mean fill")
else:
    filled_columns = []
    for col in numeric_cols:
        col_mean = df[col].mean(skipna=True)
        null_count_before = df[col].isnull().sum()
        if null_count_before > 0:
            # 填充均值
            df[col].fillna(col_mean, inplace=True)
            filled_columns.append(col)
            null_count_after = df[col].isnull().sum()

# Save data
df.to_csv(output_file, index=False)
print(f"Processing completed! The result has been saved to:{output_file}")

if unmapped_columns:
    print(f"\n unmapped_columns ({len(unmapped_columns)}):")
    for col in unmapped_columns:
        print(f"  - {col}")

if unmapped_values:
    print(f"\n unmapped_values ({len(unmapped_values)}列):")
    for col, vals in unmapped_values.items():
        print(f"  - {col}: {vals}")

if columns_to_drop:
    print(f"\ncolumns_to_drop ({len(columns_to_drop)}):")
    for col in columns_to_drop:
        print(f"  - {col}")

if filled_columns:
    print(f"\n filled_columns with mean ({len(filled_columns)}):")
    for col in filled_columns:
        col_mean = df[col].mean()
        print(f"  - {col}: mean {col_mean:.4f}")