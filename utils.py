import csv
import os

file_to_check = "nlp_papers_dataset_v6_with_isInfluential.csv"

try:
    # Use 'with open' to ensure the file is automatically closed.
    with open(file_to_check, 'r', newline='') as csvfile:
        # Create a csv.reader object to iterate over lines in the CSV.
        csv_reader = csv.reader(csvfile)

        # Use a generator expression `(1 for row in csv_reader)` inside sum().
        # This is a highly memory-efficient way to count rows. It iterates
        # through the file line by line without storing the whole file in a list.
        row_count = sum(1 for row in csv_reader)

        # Print the final count.
        # Note: This count includes the header row.
        print(f"The file '{file_to_check}' has {row_count} rows (including the header).")

        # To get the count of data rows only (excluding the header), subtract 1.
        if row_count > 0:
            data_row_count = row_count - 1
            print(f"It has {data_row_count} data rows (excluding the header).")

except FileNotFoundError:
    # This block runs if the specified file does not exist.
    print(f"Error: The file '{file_to_check}' was not found.")
except Exception as e:
    # This handles other potential errors during file processing.
    print(f"An error occurred: {e}")
finally:
    # --- 3. Clean up the created file (optional) ---
    # This removes the sample CSV file after the script is done.
    if os.path.exists(file_to_check):
        os.remove(file_to_check)
        print("-" * 20)
        print(f"Cleaned up and removed '{file_to_check}'.")