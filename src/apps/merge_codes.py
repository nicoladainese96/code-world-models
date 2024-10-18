import argparse
import json
import numpy as np
import os

def combine_codes(root_dir, combined_name="all_codes.json"):
    result_files = os.listdir(root_dir)
    tmp_codes = {}
   
    # load the results and combine them
    for r_file in result_files:
        path = os.path.join(root_dir, r_file)
        if "bleu" in path:
            continue
        elif "results.json" in path:
            continue
        elif "codes" in path and combined_name not in path:
            with open(path, "r") as f:
                results = json.load(f)
            for res in results:
                assert isinstance(results[res],list), "Problem in combining the results!"
                tmp_codes[res] = results[res]
            continue
    
    print("Amount of problems combined:", len(tmp_codes))
    with open(os.path.join(root_dir, combined_name), 'w') as f:
        json.dump(tmp_codes, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./results", type=str, help="which folder to merge the results")
    parser.add_argument("-s","--save", default="all_codes.json", type=str, help="Large final save file name. Note other files use the default value.")
    args = parser.parse_args()

    combine_codes(args.root, args.save)

if __name__ == "__main__":
    main()
