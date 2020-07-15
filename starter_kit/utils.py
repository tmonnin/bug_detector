import os
import torch
import numpy as np
from collections import defaultdict
#from torch_geometric.data import Data

from model import Net

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
        self.condition = condition # extract(if_ast["test"]["loc"], code)
        self.bin_tree = BinTree(if_ast["test"])

def generate_data_dict_sequence(d, token_embedding):
    #d = {'if_ast': d[0], 'label': d[1]}
    conditional_handler = ConditionalHandler(None, None, d["if_ast"])
    type_int_lst = []
    property_embedding_lst = []
    conditional_handler.bin_tree.to_sequence(type_int_lst, property_embedding_lst, token_embedding)
    code_adjacent_emb_lst = []
    for code_line in d["code_adjacent"]:
        code_adjacent_emb_lst.append(token_embedding[code_line])
    assert len(code_adjacent_emb_lst) == 10
    data_dict = {'type_int_lst': type_int_lst, 'property_emb_lst': property_embedding_lst, 'code_adjacent_emb_lst': code_adjacent_emb_lst}
    if "label" in d.keys():
        data_dict['label'] = torch.tensor([d["label"]], dtype=torch.int)

    return data_dict

def generate_data_dict_flattened(if_ast, token_embedding, y=None):
    conditional_handler = ConditionalHandler(None, None, if_ast)
    x_lst = []
    edge_lst = []
    # conditional_handler.bin_tree.to_list(x_lst, edge_lst, token_embedding)
    conditional_handler.bin_tree.to_flattened(x_lst, edge_lst, token_embedding, depth=4)
    type_oh = torch.zeros([15], dtype=torch.int64)  # [b, c, h, w]
    property_ft = torch.zeros((100, 1, 15), dtype=torch.float)  # [b, c, h, w]

    for i in range(len(x_lst)):
        type_oh[i] = x_lst[i][0]
        # type_oh[:,0,i] = x_lst[i][0]
        property_ft[:, 0, i] = x_lst[i][1]

    # x = torch.tensor(x_lst, dtype=torch.float)
    edge_index = None  # torch.tensor(edge_lst, dtype=torch.long)

    # if len(x) < 5:
    #    continue
    # x=x.unsqueeze(0).unsqueeze(0)
    data = None #Data(x=type_oh, edge_index=edge_index)
    data_dict = {'type_oh': type_oh, 'property_ft': property_ft, 'data': data}
    label = -1
    if y is not None:
        label = torch.tensor([y], dtype=torch.float32)
    data_dict['label'] = label

    return data_dict


class BinTree:

    def __init__(self, ast):
        if ast is None:
            # TODO remove
            print("NOT POSSIBLE")
            return
        self.ast = ast
        self.left = None
        self.right = None
        self.property = None

        #try:
        #    self.type = ast["type"]
        #except:
        #    print(ast)
        #    raise
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
            self.right=None#BinTree(ast["arguments"][0])
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
        #elif ObjectExpression
        #elif RegExp
        #elif ArrayExpression
        #elif SequenceExpression
        else:
            self.type=12
        #    print(ast)
        #    print(ast.keys())
        #    print(ast["type"])
        #    print("MISSED")

    def to_list(self, x_lst, edge_lst, token_embedding):
        idx = len(x_lst)
        #X = token_embedding[self.type]
        target = self.type
        X = torch.zeros(1, 13)
        X[range(X.shape[0]), target] = 1
        x_lst.append(X)
        if self.left is not None:
            idx_left = self.left.to_list(x_lst, edge_lst, token_embedding)
            edge_lst.append([idx, idx_left])
            edge_lst.append([idx_left, idx])
        if self.right is not None:
            idx_right = self.right.to_list(x_lst, edge_lst, token_embedding)
            edge_lst.append([idx, idx_right])
            edge_lst.append([idx_right, idx])
        return idx

    def to_sequence(self, type_int_lst, property_embedding_lst, token_embedding):
        property_embedding = torch.tensor(token_embedding[str(self.property)])
        if self.left is not None:
            self.left.to_sequence(type_int_lst, property_embedding_lst, token_embedding)
        type_int_lst.append(torch.tensor(self.type))
        property_embedding_lst.append(property_embedding)
        if self.right is not None:
            self.right.to_sequence(type_int_lst, property_embedding_lst, token_embedding)

    def to_flattened(self, x_lst, edge_lst, token_embedding, depth):
        if depth == 0:
            return None
        idx = len(x_lst)
        property_ft = torch.tensor(token_embedding[str(self.property)])
        target = self.type

        if self.left is not None:
            idx_left = self.left.to_flattened(x_lst, edge_lst, token_embedding, depth-1)
            edge_lst.append([idx, idx_left])
            edge_lst.append([idx_left, idx])
        else:
            for i in range(2**(depth-1)-1):
                x_lst.append([0, torch.zeros(1,100)])
        #type_oh = torch.zeros(13, dtype=torch.int)
        #type_oh[range(type_oh.shape[0]), target] = 1
        x_lst.append([self.type, property_ft])
        if self.right is not None:
            idx_right = self.right.to_flattened(x_lst, edge_lst, token_embedding, depth-1)
            edge_lst.append([idx, idx_right])
            edge_lst.append([idx_right, idx])
        else:
            for i in range(2**(depth-1)-1):
                x_lst.append([0, torch.zeros(1,100)])
        return idx

def load_model(model_path):
    net = Net()
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
        lines = code.splitlines()[max(start_l-1-padding,0):end_l+padding]
    else:
        lines = [""] * (padding * 2)
        lines_above = code.splitlines()[max(start_l-1-padding,0):start_l-1]
        for i, line_above in enumerate(lines_above[::-1]):
            lines[padding - 1 - i] = line_above
        lines_below = code.splitlines()[end_l:end_l+padding]
        for i, line_below in enumerate(lines_below):
            lines[padding + i] = line_below

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

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                dict_visitor(v, json_dict, expressions, identifiers)
            elif isinstance(v, list):
                for i in v:
                    dict_visitor(i, json_dict, expressions, identifiers)
            elif k == "type":
                if v == "IfStatement" and json_dict is not None: # TODO change to constants
                    # Found an IfStatement
                    json_dict[KEY_IF_AST].append(value)
                    json_dict[KEY_START_LINE].append(value["loc"]["start"]["line"])
                    # json_dict[KEY_START_LINE].append([value["loc"]["start"]["line"], value["loc"]["end"]["line"]])
                    if expressions is not None:
                        condition = value["test"]
                        type = condition["type"]
                        if type in expressions.keys():
                            expressions[type].update(list(condition.keys()))
                        else:
                            expressions[type] = set(list(condition.keys()))
                elif v == "Identifier" and "name" in value.keys(): # TODO change to constants
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

def create_result(json_dict):
    predicted_results = defaultdict(list)
    for path, d in json_dict.items():
        for i in range(len(d[KEY_IS_BUG])):
            line_begin = d[KEY_IF_AST][i]['test']['loc']['start']['line']
            line_end = d[KEY_IF_AST][i]['test']['loc']['end']['line']
            if d[KEY_IS_BUG][i]:
                for line in range(line_begin, line_end + 1): # TODO check if all lines of if statement are relevant
                    predicted_results[path].append(line)
    return dict(predicted_results)

def weighted_distribution(labels, distribution):
    class_sample_count = np.unique(labels, return_counts=True)
    print(class_sample_count)
    assert len(class_sample_count[1]) == 6
    class_p = []
    for p, n_i in zip(distribution, class_sample_count[1]):
        class_p.append(p/n_i)
    print(class_p)
    weights = []
    for label in labels:
        weights.append(class_p[label])

    return weights