import os
import torch

#from torch_geometric.data import Data

from model import Net

class ConditionalHandler:
    def __init__(self, code, condition, if_ast):
        self.code = code
        self.condition = condition # extract(if_ast["test"]["loc"], code)
        self.bin_tree = BinTree(if_ast["test"])

def generate_data_dict(if_ast, token_embedding, y=None):
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
            self.left=BinTree(ast["callee"])
            self.right=None#BinTree(ast["arguments"][0])
        elif ast["type"] == "MemberExpression":
            self.type=4
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
        elif ast["type"] == "FunctionExpression":
            self.type=10
        elif ast["type"] == "NewExpression":
            self.type=11
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

    def to_tokens(self, x_lst, edge_lst, token_embedding):

        idx = len(x_lst)
        property_ft = torch.tensor(token_embedding[str(self.property)])
        target = self.type

        if self.left is not None:
            idx_left = self.left.to_tokens(x_lst, edge_lst, token_embedding)
            edge_lst.append([idx, idx_left])
            edge_lst.append([idx_left, idx])
        else:
            for i in range(2**(depth-1)-1):
                x_lst.append([0, torch.zeros(1,100)])
        #type_oh = torch.zeros(13, dtype=torch.int)
        #type_oh[range(type_oh.shape[0]), target] = 1
        x_lst.append([self.type, property_ft])
        if self.right is not None:
            idx_right = self.right.to_tokens(x_lst, edge_lst, token_embedding)
            edge_lst.append([idx, idx_right])
            edge_lst.append([idx_right, idx])
        else:
            for i in range(2**(depth-1)-1):
                x_lst.append([0, torch.zeros(1,100)])
        return idx

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

def extract(loc_dict, code):
    start_l = loc_dict["start"]["line"]
    start_c = loc_dict["start"]["column"]
    end_l = loc_dict["end"]["line"]
    end_c = loc_dict["end"]["column"]
    lines = code.splitlines()[start_l-1:end_l]
    lines[-1] = lines[-1][:end_c]
    lines[0] = lines[0][start_c:]
    res = lines[0]
    for l in lines[1:]:
        res += " " + l
    return res