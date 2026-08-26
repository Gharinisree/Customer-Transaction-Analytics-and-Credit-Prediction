import sqlite3
import pandas as pd

def seed_sqlite():
    # Read past CSV file
    csv_file = "past_customer_transactions.csv"
    
    try:
        df = pd.read_csv(csv_file)
        # Standardize column headers (lowercase and underscore-separated)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        
        # Connect to local SQLite database (creates transactions.db if it doesn't exist)
        conn = sqlite3.connect("transactions.db")
        
        # Write CSV records into SQLite table 'customer_transactions'
        df.to_sql("customer_transactions", conn, if_exists="replace", index=False)
        conn.close()
        
        print("✅ All past CSV transactions successfully saved to local SQLite database (transactions.db)!")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{csv_file}'. Ensure the CSV file exists in your workspace root.")
    except Exception as e:
        print(f"❌ An error occurred while seeding SQLite: {e}")

if __name__ == "__main__":
    seed_sqlite()