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
    for path in list_of_json_file_paths[0:100]:
        try:
            logging.debug(path)
            j = run_bug_finding.read_json_file(path)

            code = j[run_bug_finding.KEY_CODE]
            print(code)
            print(path)
            ast = j[run_bug_finding.KEY_AST]
            json_dict[path] = defaultdict(list)
            run_bug_finding.dict_visitor_(ast, json_dict[path])
            for if_ast in json_dict[path][run_bug_finding.KEY_IF_AST]:
                incorrectly_ordered_boolean(if_ast, code)
                #condition = extract(if_ast["test"]["loc"], code)
                #print(condition)

            #res_path = os.path.splitext(os.path.join(output_dir, os.path.basename(path)))
            #res_path = res_path[0] + "_augmented" + res_path[1]
            #with open(res_path, 'w') as out_file:
            #    json.dump(j, out_file)

        except Exception as e:
            logging.error("Exception in file " + path)
            logging.error(e)
            raise e


def incomplete_conditional(if_ast: dict):
    if if_ast["test"]["type"] == "LogicalExpression":
        # if_ast["test"] = if_ast["test"]["left"]
        if_ast["test"] = if_ast["test"]["right"]


def incorrectly_ordered_boolean(if_ast: dict, code):
    if if_ast["test"]["type"] == "LogicalExpression" and if_ast["test"]["operator"] == "&&":
        code_left = extract(if_ast["test"]["left"]["loc"], code)
        code_right = extract(if_ast["test"]["right"]["loc"], code)
        if code_left in code_right:
            tmp = if_ast["test"]["left"]
            if_ast["test"]["left"] = if_ast["test"]["right"]
            if_ast["test"]["right"] = tmp



def extract(loc_dict, code):
    start_l = loc_dict["start"]["line"]
    start_c = loc_dict["start"]["column"]
    end_l = loc_dict["end"]["line"]
    end_c = loc_dict["end"]["column"]
    lines = code.splitlines()[start_l-1:end_l]
    if(len(lines) > 1):
        print("HI")
    lines[-1] = lines[-1][:end_c]
    lines[0] = lines[0][start_c:]
    res = lines[0]
    for l in lines[1:]:
        res += " " + l
    return res


def run() -> None:
    if len(sys.argv) != 3:
        print("Wrong arguments provided")
        return
    else:
        input_dir = sys.argv[1]
        print("Input dir: " + input_dir)
        output_dir = sys.argv[2]
        print("Output dir: " + output_dir)
        augment(input_dir, output_dir)

if __name__ == '__main__':
    run()