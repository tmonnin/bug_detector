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

### Pre-defined
KEY_SOURCE = 'raw_source_code'
KEY_AST = 'ast'
KEY_TOKENS = 'tokenList'
KEY_TOKENRANGE = 'tokenRangesList'

### Self-defined
KEY_START_LINE = 'start_line'
KEY_IF_AST = 'if_ast'

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

    json_dict = defaultdict(dict)
    expressions = {}
    log_level = logging.INFO
    logging.basicConfig(level=log_level)

    #if log_level == logging.DEBUG:
    list_of_json_file_paths = list_of_json_file_paths#[0:50]
    for path in list_of_json_file_paths:
        try:
            logging.debug(path)
            j = read_json_file(path)
            # print(j.keys())
            # dict_keys(['tokenList', 'raw_source_code', 'ast', 'tokenRangesList'])

            #logging.debug("Source")
            #logging.debug(j[KEY_SOURCE])
            json_dict[path] = defaultdict(list)
            dict_visitor(j[KEY_AST], json_dict[path], expressions)
            #print_expressions(expressions)
            logging.debug(json_dict[path][KEY_START_LINE])

        except Exception as e:
            logging.error("Exception in file " + path)
            logging.error(e)


    result_dict = create_result(json_dict)
    return result_dict

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
    print("AST", j[KEY_AST])
    print("Source", j[KEY_SOURCE])
    print("TokenList", j[KEY_TOKENS])
    print("TokenRange", j[KEY_TOKENRANGE])


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


def dict_visitor(value, json_dict, expressions):

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                dict_visitor(v, json_dict, expressions)
            elif isinstance(v, list):
                for i in v:
                    dict_visitor(i, json_dict, expressions)
            elif k == "type" and v == "IfStatement":
                # Found an IfStatement
                json_dict[KEY_IF_AST].append(value)
                json_dict[KEY_START_LINE].append(value["loc"]["start"]["line"])
                condition = value["test"]
                type = condition["type"]
                if type in expressions.keys():
                    expressions[type].update(list(condition.keys()))
                else:
                    expressions[type] = set(list(condition.keys()))

    elif isinstance(value, list):
        for i in value:
            dict_visitor(i, json_dict, expressions)


def dict_visitor_(value, json_dict):

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                dict_visitor_(v, json_dict)
            elif isinstance(v, list):
                for i in v:
                    dict_visitor_(i, json_dict)
            elif k == "type" and v == "IfStatement":
                # Found an IfStatement
                json_dict[KEY_IF_AST].append(value)
                json_dict[KEY_START_LINE].append(value["loc"]["start"]["line"])

    elif isinstance(value, list):
        for i in value:
            dict_visitor_(i, json_dict)

def create_result(json_dict):
    predicted_results = defaultdict(list)
    lens = [len(d[KEY_START_LINE]) for d in json_dict.values()]
    num_ifs = sum(lens)
    print("Number of if statements:", str(num_ifs))

    i=int(num_ifs/4)
    for path, d in json_dict.items():
        predicted_results[path] = d[KEY_START_LINE][:i]
        i = i - len(d[KEY_START_LINE])
        if i <= 0:
            break

    return dict(predicted_results)