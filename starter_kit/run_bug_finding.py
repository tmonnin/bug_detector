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
import utils



model_path = "model"

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
        print("Error loading JSON " + json_file_path)
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
    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path_full = os.path.join(cur_dir_path, model_path)
    net = utils.load_model(model_path_full)

    #if log_level == logging.DEBUG:
    list_of_json_file_paths = list_of_json_file_paths#[0:50]
    for path in list_of_json_file_paths:
        try:
            logging.debug(path)
            j = read_json_file(path)
            # print(j.keys())
            # dict_keys(['tokenList', 'raw_source_code', 'ast', 'tokenRangesList'])

            logging.debug("Code")
            logging.debug(j[utils.KEY_CODE])
            json_dict[path] = defaultdict(list)
            utils.dict_visitor(j[utils.KEY_AST], json_dict[path])

            for if_ast in json_dict[path][utils.KEY_IF_AST]:
                data_dict = utils.generate_data_dict_sequence(if_ast, token_embedding)
                is_bug = net.classify([data_dict])
                json_dict[path][utils.KEY_IS_BUG] += is_bug

            #utils.print_expressions(expressions)
            logging.debug(json_dict[path][utils.KEY_START_LINE])

        except Exception as e:
            logging.error("Exception in file " + path)
            logging.error(e)
            import traceback
            traceback.print_exc()

    result_dict = utils.create_result(json_dict)
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


