from typing import List, Dict
import os
import fasttext
from collections import defaultdict
import traceback
import logging

import utils
import train


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
    log_level = logging.INFO
    logging.basicConfig(level=log_level)
    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path_full = os.path.join(cur_dir_path, train.model_path)
    net = utils.load_model(model_path_full, train.strategy)

    list_of_json_file_paths = list_of_json_file_paths[:]
    for path in list_of_json_file_paths:
        try:
            logging.debug(path)
            if_dict_lst, code, code_identifier_lst = utils.extract_if_dicts(path)
            json_dict[path] = defaultdict(list)
            for if_dict in if_dict_lst:
                if train.strategy == 'lstm':
                    d = utils.generate_data_dict_sequence(if_dict, token_embedding)
                elif train.strategy == 'gcn':
                    d = utils.generate_data_dict_graph(if_dict, token_embedding)
                elif train.strategy == 'cnn':
                    d = utils.generate_data_dict_flattened(if_dict, token_embedding)
                is_bug = net.classify([d])
                json_dict[path][utils.KEY_IF_AST] += [if_dict[utils.KEY_IF_AST]]
                json_dict[path][utils.KEY_IS_BUG] += is_bug

            logging.debug(json_dict[path][utils.KEY_START_LINE])

        except Exception as e:
            logging.error("Exception in file " + path)
            logging.error(e)
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


