import pandas as pd
import argparse

def process_tsv(input_file, output_file):
    df = pd.read_csv(input_file, sep='\t')

    df['segment_id'] = df['segment_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    df_unique_segment_id = df['segment_id'].drop_duplicates()

    df_unique_segment_id.to_csv(output_file, sep='\t', index=False, header=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_file', type=str, required=True, help="input tsv file")
    
    parser.add_argument('-o', '--output_file', type=str, default='output_segment_ids.tsv', help="output tsv file")

    args = parser.parse_args()

    process_tsv(args.input_file, args.output_file)


