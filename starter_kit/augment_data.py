import run_bug_finding
import logging
from pathlib import Path
import sys
from collections import defaultdict
import json
import os

logging.basicConfig(level=logging.INFO)

def augment(input_dir, output_dir):
    list_of_json_file_paths = list(Path(input_dir).glob('**/*.json'))
    list_of_json_file_paths = [str(p) for p in list_of_json_file_paths]
    logging.info(str(len(list_of_json_file_paths)) + " JSON files found")
    json_dict = defaultdict(dict)
    for path in list_of_json_file_paths[0:1]:
        try:
            logging.debug(path)
            j = run_bug_finding.read_json_file(path)
            # print(j.keys())
            # dict_keys(['tokenList', 'raw_source_code', 'ast', 'tokenRangesList'])

            # logging.debug("Source")
            # logging.debug(j[KEY_SOURCE])
            ast = j[run_bug_finding.KEY_AST]
            json_dict[path] = defaultdict(list)
            run_bug_finding.dict_visitor_(ast, json_dict[path])
            # print_expressions(expressions)
            logging.debug(json_dict[path][run_bug_finding.KEY_START_LINE])
            res_path = os.path.splitext(os.path.join(output_dir, os.path.basename(path)))
            res_path = res_path[0] + "_augmented" + res_path[1]
            with open(res_path, 'w') as out_file:
                json.dump(j, out_file)

        except Exception as e:
            logging.error("Exception in file " + path)
            logging.error(e)
            raise e

def run() -> None:
    if len(sys.argv) != 3:  # Use the default input and output directories if no arguments are provided
        return
    else:
        input_dir = sys.argv[1]
        print("Input dir: " + input_dir)
        output_dir = sys.argv[2]
        print("Output dir: " + output_dir)
        augment(input_dir, output_dir)

if __name__ == '__main__':
    run()