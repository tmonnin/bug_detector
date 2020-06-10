from typing import List, Dict, Union, DefaultDict
from pathlib import Path
import torch  # -> version --> 1.5.0
import numpy as np
import os
import fasttext
from collections import defaultdict
import json
import codecs
import logging


def read_json_file(json_file_path: str) -> List:
    """ Read a JSON file given path """
    try:
        obj_text = codecs.open(json_file_path, 'r',
                               encoding='utf-8').read()
        return json.loads(obj_text)
    except FileNotFoundError:
        print(
            "File {} not found. Please provide a correct file path Eg. ./results/hello.json".format(json_file_path))
        return []
    except Exception as e:
        # Most likely malformed JSON file
        return []


def find_bugs_in_js_files(list_of_json_file_paths: List[str], token_embedding: fasttext.FastText) -> Dict[str, List[int]]:
    r"""
    Please DO NOT delete the 'metadata' file in the current directory else your submission will not be scored on codalab.

    :param list_of_json_file_paths:
        Example:
            list_of_json_file_paths = [
                'dataset/1.json',
                'dataset/2.json',
                'dataset/3.json',
                'dataset/4.json',
            ]
    :param token_embedding: get embedding for tokens. The pre-trained embeddings have been learned using fastText (https://fasttext.cc/docs/en/support.html)
        Example:
            token_embedding['foo'] # Gets vector representation for the Identifier 'foo'
            token_embedding['true'] # Gets vector representation for the 'true
    :return: A dictionary of the found bugs in the given list of JSON files. The keys should be the file paths and the corresponding values should be list of line numbers where the bug occurs. The format of the dict should be returned as follows:
            {
                'dataset/1.json': [1, 2],
                'dataset/2.json': [11],
                'dataset/3.json': [6],
                'dataset/4.json': [4, 2]
            }
    """

    #####################################################
    #                                                   #
    #   1. Write your code below.                       #
    #   2. You may use the read_json_file() helper      #
    #      function to read a JSON file.                #
    #   3. Return a dict with the found bugs from here. #
    #                                                   #
    #####################################################

    # For each file, the current naive implementation returns a random line number between 1-500
    # Replace this with your own code

    if_dict = {}
    len_list = []
    expressions = {}
    log_level = logging.INFO
    logging.basicConfig(level=log_level)

    def dict_visitor(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    dict_visitor(v)
                elif isinstance(v, list):
                    for i in v:
                        dict_visitor(i)
                else:
                    #print(k, v)
                    if k == "type" and v == "IfStatement":
                        loc_lst.append(d["loc"])
                        #print("Check")
                        condition = d["test"]
                        # d["test"]["object"]
                        # d["test"]["property"]
                        #print(condition)
                        type = condition["type"]
                        if type in expressions.keys():
                            expressions[type].update(list(condition.keys()))
                        else:
                            expressions[type] = set(list(condition.keys()))

        elif isinstance(d, list):
            for i in d:
                dict_visitor(i)
        else:
            pass
            #print(type(d), d)

    #if log_level == logging.DEBUG:
    list_of_json_file_paths = list_of_json_file_paths#[0:100]
    for path in list_of_json_file_paths:
        try:
            logging.debug(path)
            j = read_json_file(path)
            #file = open(path)
            #j = json.load(file)
            # print(j.keys())
            # dict_keys(['tokenList', 'raw_source_code', 'ast', 'tokenRangesList'])

            token_list = j["tokenList"]
            indices = [i for i, x in enumerate(token_list) if x == "if"]
            len_list.append(len(indices))

            #logging.debug("Source")
            #logging.debug(j["raw_source_code"])
            ast = j["ast"]
            loc_lst = []
            dict_visitor(ast)
            logging.debug(loc_lst)
            if_dict[path] = []
            for loc in loc_lst:
                if_dict[path].append(loc["start"]["line"])

        except Exception as e:
            logging.error("Exception in file " + path)
            logging.error(e)

    predicted_results = defaultdict(list)

    for path, start_line in if_dict.items():
        predicted_results[path] = start_line

    return dict(predicted_results)

# TODO
# Remove comments ?
# Find if token - multiple ones
# Create training data set
# Encode AST
# Easy approach: Input Expression type, output binary
# Check all 5 bug cases and derive analysis approach
# Create RNN for Syntax tree input
# Define binary cross entropy loss


def print_json(j):
    print("AST", j["ast"])
    print("Source", j["raw_source_code"])
    print("TokenList", j["tokenList"])
    print(j["tokenRangesList"])

def count_unique(list):
    values, counts = np.unique(list, return_counts=True)
    print(values)
    print(counts)

def print_expressions(expressions):
    for k, v in expressions.items():
        print(k, v)
    # UnaryExpression
    # {'range', 'prefix', 'operator', 'type', 'argument', 'loc'}
    # BinaryExpression
    # {'range', 'right', 'operator', 'type', 'loc', 'left'}
    # CallExpression
    # {'callee', 'type', 'range', 'loc', 'arguments'}
    # MemberExpression
    # {'computed', 'property', 'range', 'type', 'loc', 'object'}
    # Literal
    # {'raw', 'value', 'range', 'loc', 'type'}
    # AssignmentExpression
    # {'range', 'right', 'operator', 'type', 'loc', 'left'}
    # LogicalExpression
    # {'range', 'right', 'operator', 'type', 'loc', 'left'}
    # Identifier
    # {'name', 'type', 'range', 'loc'}