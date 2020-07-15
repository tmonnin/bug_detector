import run_bug_finding
import logging
import random
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
    for index, path in enumerate(list_of_json_file_paths):#[0:10000]:
        try:
            res_path = os.path.splitext(os.path.join(output_dir, os.path.basename(path)))
            if not os.path.exists(res_path[0] + "_0" + res_path[1]):
                logging.info(path)
                j = run_bug_finding.read_json_file(path)

                code = j[utils.KEY_CODE]
                logging.debug("Code: " + str(code))
                ast = j[utils.KEY_AST]
                token = j[utils.KEY_TOKENS]
                token_range = j[utils.KEY_TOKENRANGE]

                json_dict[path] = defaultdict(list)
                code_identifier_lst = []
                utils.dict_visitor(ast, json_dict[path], identifiers=code_identifier_lst)
                random.shuffle(code_identifier_lst)
                for if_ast in json_dict[path][utils.KEY_IF_AST]:
                    condition = utils.extract(if_ast["test"]["loc"], code)
                    #print(condition)
                    # Skip test conditions that are too large TODO senseful?
                    if len(condition) > 100:
                        continue

                    code_adjacent = utils.extract(if_ast["test"]["loc"], code, padding=5, skip_condition=True, return_list=True)
                    #print(code_adjacent)
                    logging.debug("Condition: " + str(condition))
                    funcs = [(identity, 0), (incomplete_conditional_left, 1), (incomplete_conditional_right, 1),
                             (incorrectly_ordered_boolean, 2), (wrong_identifier, 3), (negated_condition_remove, 4),
                             (negated_condition_add, 4), (wrong_operator, 5)]
                    for aug_function, label in funcs:
                        if_ast_copy = deepcopy(if_ast)
                        is_augmented = aug_function(if_ast_copy, code, code_identifier_lst)
                        if is_augmented:
                            d = {'if_ast': if_ast_copy, 'condition': condition, 'code_adjacent': code_adjacent, 'label': label}
                            res_dict[path].append(d)
                            logging.debug("Augmented: " + str(aug_function))
                            aug_count_dict[str(label)] += 1

                for i, res in enumerate(res_dict[path]):
                    output_path = res_path[0] + "_" + str(i) + res_path[1]
                    with open(output_path, 'w') as out_file:
                        json.dump(res_dict[path][i], out_file)

        except Exception as e:
            logging.error("Exception in file " + path)
            logging.error(e)
            raise e

        logging.info(aug_count_dict.items())
        logging.info(str(index) + "/" + str(len(list_of_json_file_paths)))

def identity(if_ast: dict, code, code_identifier_lst):
    return True

def incomplete_conditional_left(if_ast: dict, code, code_identifier_lst):
    if if_ast["test"]["type"] == "LogicalExpression":
        if_ast["test"] = if_ast["test"]["left"]
        return True

def incomplete_conditional_right(if_ast: dict, code, code_identifier_lst):
    if if_ast["test"]["type"] == "LogicalExpression":
        if_ast["test"] = if_ast["test"]["right"]
        return True

def incorrectly_ordered_boolean(if_ast: dict, code, code_identifier_lst):
    if if_ast["test"]["type"] == "LogicalExpression" and if_ast["test"]["operator"] == "&&":
        code_left = utils.extract(if_ast["test"]["left"]["loc"], code)
        code_right = utils.extract(if_ast["test"]["right"]["loc"], code)
        if code_left in code_right: # TODO similarity
            tmp = if_ast["test"]["left"]
            if_ast["test"]["left"] = if_ast["test"]["right"]
            if_ast["test"]["right"] = tmp
            return True

def wrong_identifier(if_ast: dict, code, code_identifier_lst):
    code_condition_padded = utils.extract(if_ast["test"]["loc"], code, padding=5)
    condition_identifier_lst = []
    utils.dict_visitor(if_ast, identifiers=condition_identifier_lst)
    if len(condition_identifier_lst):
        identifier_to_augment = random.choice(condition_identifier_lst)
        # TODO identifier must stand alone?
        for identifier in code_identifier_lst:
            identifier_start = identifier["loc"]["start"]["line"]
            augment_start = identifier_to_augment["loc"]["start"]["line"]
            if identifier_start < (augment_start - 5) and identifier["name"] not in code_condition_padded:
                # TODO choose most similar identifier
                # TODO near neighborhood could be feasible
                identifier_to_augment["name"] = identifier["name"]
                return True

def negated_condition_remove(if_ast: dict, code, code_identifier_lst):
    if if_ast["test"]["type"] == "UnaryExpression" and if_ast["test"]["operator"] == "!":
        # Remove negation operator
        if_ast["test"] = if_ast["test"]["argument"]
        return True

def negated_condition_add(if_ast: dict, code, code_identifier_lst):
    # Choose types where a negated condition can easily be added without thinking about brackets
    if if_ast["test"]["type"] in ("MemberExpression", "Identifier", "CallExpression"):
        arg = if_ast["test"]
        # Add negation operator
        if_ast["test"] = {
            "type": "UnaryExpression",
            "operator": "!",
            "prefix": True,
            "argument": arg
        }
        return True

def wrong_operator_(if_ast: dict, code, code_identifier_lst):
    if if_ast["test"]["type"] == "BinaryExpression":
        if if_ast["test"]["operator"] == "<":
            if_ast["test"]["operator"] = ">"
            return True
        elif if_ast["test"]["operator"] == ">":
            if_ast["test"]["operator"] = "<"
            return True
        # TODO equals, shift operator

def wrong_operator(if_ast: dict, code, code_identifier_lst):
    if if_ast["test"]["type"] == "BinaryExpression":
        if if_ast["test"]["operator"] == "<=":
            if_ast["test"]["operator"] = "<"
            return True
        elif if_ast["test"]["operator"] == ">=":
            if_ast["test"]["operator"] = ">"
            return True
        elif if_ast["test"]["operator"] == "<":
            if_ast["test"]["operator"] = "<="
            return True
        elif if_ast["test"]["operator"] == ">":
            if_ast["test"]["operator"] = ">="
            return True
        elif if_ast["test"]["operator"] == "===":
            if_ast["test"]["operator"] = "=="
            return True
        elif if_ast["test"]["operator"] == "==":
            if_ast["test"]["operator"] = "="
            if_ast["test"]["type"] = "AssignmentExpression"
            return True
        elif if_ast["test"]["operator"] == "!==":
            if_ast["test"]["operator"] = "!="
            if_ast["test"]["type"] = "AssignmentExpression"
            return True
    elif if_ast["test"]["type"] == "LogicalExpression":
        if if_ast["test"]["operator"] == "&&":
            if_ast["test"]["operator"] = "&"
            if_ast["test"]["type"] = "BinaryExpression"
            return True
        if if_ast["test"]["operator"] == "||":
            if_ast["test"]["operator"] = "|"
            if_ast["test"]["type"] = "BinaryExpression"
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
