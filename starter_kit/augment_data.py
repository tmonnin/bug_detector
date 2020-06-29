import run_bug_finding
import logging
from pathlib import Path
import sys
from collections import defaultdict
from copy import deepcopy
import json
import os
import utils

logging.basicConfig(level=logging.INFO)


def augment(input_dir, output_dir):
    list_of_json_file_paths = list(Path(input_dir).glob('**/*.json'))
    list_of_json_file_paths = [str(p) for p in list_of_json_file_paths]
    logging.info(str(len(list_of_json_file_paths)) + " JSON files found")
    json_dict = defaultdict(dict)
    res_dict = defaultdict(list)
    aug_count_dict = defaultdict(lambda: 0)
    for path in list_of_json_file_paths[0:1000]:
        try:
            res_path = os.path.splitext(os.path.join(output_dir, os.path.basename(path)))
            if not os.path.exists(res_path[0] + "_0" + res_path[1]):
                logging.info(path)
                j = run_bug_finding.read_json_file(path)

                code = j[run_bug_finding.KEY_CODE]
                logging.debug("Code: " + str(code))
                ast = j[run_bug_finding.KEY_AST]
                json_dict[path] = defaultdict(list)
                run_bug_finding.dict_visitor_(ast, json_dict[path])
                for if_ast in json_dict[path][run_bug_finding.KEY_IF_AST]:
                    condition = utils.extract(if_ast["test"]["loc"], code)
                    logging.info("Condition: " + str(condition))
                    one_augmented = False
                    for aug_function in [incomplete_conditional, incorrectly_ordered_boolean, wrong_identifier, negated_condition, wrong_operator]:
                        if_ast_copy = deepcopy(if_ast)
                        is_augmented = aug_function(if_ast_copy, code)
                        if is_augmented:
                            res_dict[path].append([if_ast_copy, 1])
                            logging.info("Augmented: " + str(aug_function))
                            aug_count_dict[aug_function] += 1
                            one_augmented = True
                        #bin_tree = utils.ConditionalHandler(code, condition, if_ast_copy)
                        #logging.debug(bin_tree)
                    if not one_augmented:
                        res_dict[path].append([if_ast, 0])
                        aug_count_dict["None"] += 1

                for i, res in enumerate(res_dict[path]):
                    output_path = res_path[0] + "_" + str(i) + res_path[1]
                    with open(output_path, 'w') as out_file:
                        json.dump(res_dict[path][i], out_file)

        except Exception as e:
            logging.error("Exception in file " + path)
            logging.error(e)
            raise e

        logging.info(aug_count_dict.items())


def incomplete_conditional(if_ast: dict, code):
    if if_ast["test"]["type"] == "LogicalExpression":
        if_ast["test"] = if_ast["test"]["left"]
        #if_ast["test"] = if_ast["test"]["right"]
        return True


def incorrectly_ordered_boolean(if_ast: dict, code):
    if if_ast["test"]["type"] == "LogicalExpression" and if_ast["test"]["operator"] == "&&":
        code_left = utils.extract(if_ast["test"]["left"]["loc"], code)
        code_right = utils.extract(if_ast["test"]["right"]["loc"], code)
        if code_left in code_right:
            tmp = if_ast["test"]["left"]
            if_ast["test"]["left"] = if_ast["test"]["right"]
            if_ast["test"]["right"] = tmp
            return True


def wrong_identifier(if_ast: dict, code):
    return None


def negated_condition(if_ast: dict, code):
    if if_ast["test"]["type"] == "UnaryExpression" and if_ast["test"]["operator"] == "!":
        # Remove negation operator
        if_ast["test"] = if_ast["test"]["argument"]
        return True
    # else:
    #     # Pack test into negation
    #     arg = if_ast["test"]
    #     if_ast["test"] = {
    #         "type": "UnaryExpression",
    #         "operator": "!",
    #         "prefix": True,
    #         "argument": arg
    #     }


def wrong_operator(if_ast: dict, code):
    if if_ast["test"]["type"] == "BinaryExpression":
        if if_ast["test"]["operator"] == "<":
            if_ast["test"]["operator"] = ">"
            return True
        elif if_ast["test"]["operator"] == ">":
            if_ast["test"]["operator"] = "<"
            return True


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
