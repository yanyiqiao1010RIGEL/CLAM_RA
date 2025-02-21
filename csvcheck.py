import pandas as pd


# Function to load CSV and svs file names from a text file
def find_unmatched_svs(csv_path, svs_txt_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Extract the slide_id without the extensions from the CSV
    csv_slide_ids = df['slide_id'].apply(lambda x: x.split('.')[0]).tolist()

    # Read the svs file names from the text file
    with open(svs_txt_path, 'r') as file:
        svs_files = [line.strip().split('.')[0] for line in file.readlines()]

    # Find the unmatched files
    unmatched_in_csv = set(svs_files) - set(csv_slide_ids)
    unmatched_in_svs = set(csv_slide_ids) - set(svs_files)

    return unmatched_in_csv, unmatched_in_svs


csv_path = '/Users/yanyiqiao/Downloads/CLAM_RA/Labels/Florida.csv'
svs_txt_path = '/Users/yanyiqiao/Downloads/CLAM_RA/Florida.txt'
unmatched_in_csv, unmatched_in_svs = find_unmatched_svs(csv_path, svs_txt_path)
print("Unmatched in CSV:", unmatched_in_csv)
print("Unmatched in SVS:", unmatched_in_svs)
