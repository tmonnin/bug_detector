import os
import torch
import numpy as np
from collections import defaultdict
from typing import List
import json
import codecs
import random
import re

from model_lstm import LSTMNet
from model_gcn import GCNNet
from model_cnn import CNNNet

### Pre-defined
KEY_CODE = 'raw_source_code'
KEY_AST = 'ast'
KEY_TOKENS = 'tokenList'
KEY_TOKENRANGE = 'tokenRangesList'

### Self-defined
KEY_START_LINE = 'start_line'
KEY_IF_AST = 'if_ast'
KEY_IS_BUG = 'is_bug'

class ConditionalHandler:
    def __init__(self, code, condition, if_ast):
        self.code = code
        self.condition = condition
        self.bin_tree = BinTree(if_ast["test"])

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

def extract_if_dicts(path):
    j = read_json_file(path)
    code = j[KEY_CODE]
    ast = j[KEY_AST]
    token = j[KEY_TOKENS]
    token_range = j[KEY_TOKENRANGE]

    json_dict = defaultdict(list)
    code_identifier_lst = []
    dict_visitor(ast, json_dict, identifiers=code_identifier_lst)
    random.shuffle(code_identifier_lst)
    if_dict_lst = []
    for if_ast in json_dict[KEY_IF_AST]:
        condition = extract(if_ast["test"]["loc"], code)
        code_adjacent = extract(if_ast["test"]["loc"], code, padding=5, skip_condition=True, return_list=True)
        d = {'if_ast': if_ast, 'condition': condition, 'code_adjacent': code_adjacent}
        if_dict_lst.append(d)

    return if_dict_lst, code, code_identifier_lst

def generate_data_dict_sequence(d, token_embedding):
    conditional_handler = ConditionalHandler(None, None, d["if_ast"])
    type_int_lst = []
    property_emb_lst = []
    conditional_handler.bin_tree.to_sequence(type_int_lst, property_emb_lst, token_embedding)
    token_lst = []
    for code_line in d["code_adjacent"][:5]:
        # Generate tokens by splitting alphabetic words
        token_lst += re.findall('[a-zA-Z]+', code_line)
    if not len(token_lst):
        # Add a 0 to have at least one token
        token_lst.append("0")
    token_embedding_lst = []
    for token in token_lst:
        token_emb = torch.tensor(token_embedding[token])
        token_embedding_lst.append(token_emb)

    data_dict = {'type_int_lst': type_int_lst, 'property_emb_lst': property_emb_lst, 'code_adjacent_emb_lst': token_embedding_lst}
    if "label" in d.keys():
        data_dict['label'] = torch.tensor([d["label"]], dtype=torch.int)

    return data_dict

def generate_data_dict_flattened(d, token_embedding, y=None):
    conditional_handler = ConditionalHandler(None, None, d["if_ast"])
    x_lst = []
    conditional_handler.bin_tree.to_flattened(x_lst, token_embedding, depth=4)
    type_int_lst = []
    property_emb_lst = []
    for i in range(len(x_lst)):
        type_int_lst.append(x_lst[i][0])
        property_emb_lst.append(x_lst[i][1])

    data_dict = {'type_int_lst': type_int_lst, 'property_emb_lst': property_emb_lst}
    if "label" in d.keys():
        data_dict['label'] = torch.tensor([d["label"]], dtype=torch.int)

    return data_dict

def generate_data_dict_graph(d, token_embedding):
    conditional_handler = ConditionalHandler(None, None, d['if_ast'])
    type_int_lst = []
    property_emb_lst = []
    edge_lst = []
    conditional_handler.bin_tree.to_graph(type_int_lst, property_emb_lst, edge_lst, token_embedding)
    data_dict = {'type_int_lst': type_int_lst, 'property_emb_lst': property_emb_lst, 'edge_lst': edge_lst}
    if "label" in d.keys():
        data_dict['label'] = torch.tensor([d["label"]], dtype=torch.int)
    return data_dict


class BinTree:

    def __init__(self, ast):
        self.ast = ast
        self.left = None
        self.right = None
        self.property = "" # Default for uncaught cases
        self.type = 0 # Default for uncaught cases
        if ast["type"] == "UnaryExpression":
            self.type=1
            self.property = ast["operator"]
            self.left = BinTree(ast["argument"])
        elif ast["type"] == "BinaryExpression":
            self.type=2
            self.property=ast["operator"]
            self.left=BinTree(ast["left"])
            self.right= BinTree(ast["right"])
        elif ast["type"] == "CallExpression":
            self.type=3
            self.property="()"
            self.left=BinTree(ast["callee"])
            self.right=None
        elif ast["type"] == "MemberExpression":
            self.type=4
            self.property="."
            self.left=BinTree(ast["object"])
            self.right=BinTree(ast["property"])
        elif ast["type"] == "Literal":
            self.type=5
            self.property=ast["raw"]
        elif ast["type"] == "AssignmentExpression":
            self.type=6
            self.property=ast["operator"]
            self.left=BinTree(ast["left"])
            self.right=BinTree(ast["right"])
        elif ast["type"] == "LogicalExpression":
            self.type=7
            self.property=ast["operator"]
            self.left=BinTree(ast["left"])
            self.right=BinTree(ast["right"])
        elif ast["type"] == "Identifier":
            self.type=8
            self.property=ast["name"]
        elif ast["type"] == "ThisExpression":
            self.type=9
            self.property="this"
        elif ast["type"] == "FunctionExpression":
            self.type=10
            try:
                self.property=ast["id"]["name"]
            except TypeError: # anonymous function
                self.property="function"
        elif ast["type"] == "NewExpression":
            self.type=11
            self.property="new"
        elif ast["type"] == "UpdateExpression":
            self.type=12
            self.property = ast["operator"]
            self.left = BinTree(ast["argument"])
        #elif ast["type"] == "ObjectExpression":
        #    self.type=13
        #elif ast["type"] == "RegExp":
        #    self.type=14
        #    self.property = "regex"
        #elif ast["type"] == "ArrayExpression":
        #    self.type=15
        #    self.property = "array"
        #elif ast["type"] == "SequenceExpression":
        #    self.type=16
        else:
            pass

    def to_graph(self, type_int_lst, property_emb_lst, edge_lst, token_embedding):
        idx = len(type_int_lst)
        property_emb = torch.tensor(token_embedding[str(self.property)])
        type_int_lst.append(torch.tensor(self.type))
        property_emb_lst.append(property_emb)
        if self.left is not None:
            idx_left = self.left.to_graph(type_int_lst, property_emb_lst, edge_lst, token_embedding)
            edge_lst.append([idx, idx_left])
            edge_lst.append([idx_left, idx])
        if self.right is not None:
            idx_right = self.right.to_graph(type_int_lst, property_emb_lst, edge_lst, token_embedding)
            edge_lst.append([idx, idx_right])
            edge_lst.append([idx_right, idx])
        if idx == 0 and not len(edge_lst):
            # if test condition is only one node in AST, add self-edges
            edge_lst.append([0, 0])
            edge_lst.append([0, 0])
        return idx

    def to_sequence(self, type_int_lst, property_emb_lst, token_embedding):
        property_emb = torch.tensor(token_embedding[str(self.property)])
        if self.left is not None:
            self.left.to_sequence(type_int_lst, property_emb_lst, token_embedding)
        type_int_lst.append(torch.tensor(self.type))
        property_emb_lst.append(property_emb)
        if self.right is not None:
            self.right.to_sequence(type_int_lst, property_emb_lst, token_embedding)

    def to_flattened(self, x_lst, token_embedding, depth):
        if depth == 0:
            return None
        type_tensor = torch.tensor([self.type], dtype=torch.int64)
        property_tensor = torch.tensor(token_embedding[str(self.property)]).unsqueeze(0)

        if self.left is not None:
            self.left.to_flattened(x_lst, token_embedding, depth-1)
        else:
            for i in range(2**(depth-1)-1):
                x_lst.append([torch.zeros(1, dtype=torch.int64), torch.zeros(1,100)])
        x_lst.append([type_tensor, property_tensor])
        if self.right is not None:
            self.right.to_flattened(x_lst, token_embedding, depth-1)
        else:
            for i in range(2**(depth-1)-1):
                x_lst.append([torch.zeros(1, dtype=torch.int64), torch.zeros(1,100)])

def load_model(model_path, strategy='lstm'):
    if strategy == 'lstm':
        net = LSTMNet()
    elif strategy == 'gcn':
        net = GCNNet()
    elif strategy == 'cnn':
        net = CNNNet()
    else:
        raise Exception("Strategy not supported")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict)
    return net

def extract(loc_dict, code, padding=0, skip_condition=False, return_list=False):
    start_l = loc_dict["start"]["line"]
    start_c = loc_dict["start"]["column"]
    end_l = loc_dict["end"]["line"]
    end_c = loc_dict["end"]["column"]
    if not skip_condition:
        # Include if statement
        lines = code.splitlines()[max(start_l-1-padding,0):end_l+padding]
    else:
        # Assert padding * 2 number of lines
        lines = [""] * (padding * 2)
        lines_above = code.splitlines()[max(start_l-1-padding,0):start_l-1]
        for i, line_above in enumerate(lines_above[::-1]):
            lines[padding - 1 - i] = line_above
        lines_below = code.splitlines()[end_l:end_l+padding]
        for i, line_below in enumerate(lines_below):
            lines[padding + i] = line_below

    # If padding is 0, extract precise character range
    if padding == 0:
        lines[-1] = lines[-1][:end_c]
        lines[0] = lines[0][start_c:]
    if not return_list:
        res = lines[0]
        for l in lines[1:]:
            res += " " + l
    else:
        res = lines
    return res

def dict_visitor(value, json_dict=None, expressions=None, identifiers=None):
    """Generic method to extract if statements and more from AST dict"""

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                dict_visitor(v, json_dict, expressions, identifiers)
            elif isinstance(v, list):
                for i in v:
                    dict_visitor(i, json_dict, expressions, identifiers)
            elif k == "type":
                if v == "IfStatement" and json_dict is not None:
                    # Found an IfStatement
                    json_dict[KEY_IF_AST].append(value)
                    json_dict[KEY_START_LINE].append(value["loc"]["start"]["line"])
                    if expressions is not None:
                        condition = value["test"]
                        type = condition["type"]
                        if type in expressions.keys():
                            expressions[type].update(list(condition.keys()))
                        else:
                            expressions[type] = set(list(condition.keys()))
                elif v == "Identifier" and "name" in value.keys():
                    if identifiers is not None:
                        identifiers.append(value)

    elif isinstance(value, list):
        for i in value:
            dict_visitor(i, json_dict, expressions, identifiers)

def print_json(j):
    print("AST", j[KEY_AST])
    print("Source", j[KEY_CODE])
    print("TokenList", j[KEY_TOKENS])
    print("TokenRange", j[KEY_TOKENRANGE])

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

def create_result(json_dict):
    predicted_results = dict()
    for path, d in json_dict.items():
        predicted_results[path] = []
        for i in range(len(d[KEY_IS_BUG])):
            if d[KEY_IS_BUG][i]:
                line_begin = d[KEY_IF_AST][i]['test']['loc']['start']['line']
                predicted_results[path].append(line_begin)
    return dict(predicted_results)

def weighted_distribution(labels, distribution):
    class_sample_count = np.unique(labels, return_counts=True)
    print("Class labels and occurences:", str(class_sample_count))
    assert len(class_sample_count[1]) == 6
    class_p = []
    for p, n_i in zip(distribution, class_sample_count[1]):
        class_p.append(p/n_i)
    weights = []
    for label in labels:
        weights.append(class_p[label])

    return weights
