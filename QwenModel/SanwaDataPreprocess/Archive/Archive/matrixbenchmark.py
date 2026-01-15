import pandas as pd
import os

# ================= USER CONFIGURATION =================
# 1. Input: The Benchmark Matrix created in the previous step
BENCHMARK_FILE = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640/Cross_Dataset_Deletion_Benchmark.csv'

# ================= PROCESSING LOGIC =================
def add_time_difference():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"❌ File not found: {BENCHMARK_FILE}")
        return

    print(f"Processing: {os.path.basename(BENCHMARK_FILE)}...")

    try:
        df = pd.read_csv(BENCHMARK_FILE)
    except Exception as e:
        print(f"  [Error] {e}")
        return

    if 'Machine_Time' not in df.columns:
        print("  [Error] 'Machine_Time' column missing. Please run the previous step first.")
        return

    # 1. Convert to Datetime
    #    Format expected: 'YYYY-MM-DD HH:MM:SS'
    df['Temp_Time'] = pd.to_datetime(df['Machine_Time'], errors='coerce')

    # 2. Sort (Just to be safe)
    df.sort_values(by='Temp_Time', inplace=True)

    # 3. Calculate Difference
    #    diff() calculates current_row - previous_row
    df['Time_Diff_Sec'] = df['Temp_Time'].diff().dt.total_seconds()

    # 4. Cleanup
    #    Fill the first row (NaN) with 0 or empty
    df['Time_Diff_Sec'] = df['Time_Diff_Sec'].fillna(0)
    df.drop(columns=['Temp_Time'], inplace=True)

    # 5. Reorder Columns
    #    Put Time_Diff_Sec right after Machine_Time (or at the end)
    cols = list(df.columns)
    if 'Time_Diff_Sec' in cols: cols.remove('Time_Diff_Sec')
    
    # Place it at the very end
    cols.append('Time_Diff_Sec')
    df = df[cols]

    # 6. Save
    df.to_csv(BENCHMARK_FILE, index=False)

    print(f"✅ Updated: {BENCHMARK_FILE}")
    print(f"   Added 'Time_Diff_Sec' column.")
    
    print("\n[Preview]")
    print(df[['Machine_Time', 'Time_Diff_Sec']].head(10))

if __name__ == "__main__":
    add_time_difference()