class ConditionalHandler:
    def __init__(self, code, condition, if_ast):
        self.code = code
        self.condition = condition # extract(if_ast["test"]["loc"], code)
        self.bin_tree = BinTree(if_ast["test"])




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

        try:
            self.type = ast["type"]
        except:
            print(ast)
            raise
        if ast["type"] == "UnaryExpression":
            self.property = ast["operator"]
            self.left = BinTree(ast["argument"])
        elif ast["type"] == "BinaryExpression":
            self.property=ast["operator"]
            self.left=BinTree(ast["left"])
            self.right= BinTree(ast["right"])
        elif ast["type"] == "CallExpression":
            self.left=BinTree(ast["callee"])
            self.right=None#BinTree(ast["arguments"][0])
        elif ast["type"] == "MemberExpression":
            self.left=BinTree(ast["object"])
            self.right=BinTree(ast["property"])
        elif ast["type"] == "Literal":
            self.property=ast["raw"]
        elif ast["type"] == "AssignmentExpression":
            self.property=ast["operator"]
            self.left=BinTree(ast["left"])
            self.right=BinTree(ast["right"])
        elif ast["type"] == "LogicalExpression":
            self.property=ast["operator"]
            self.left=BinTree(ast["left"])
            self.right=BinTree(ast["right"])
        elif ast["type"] == "Identifier":
            self.property=ast["name"]
        #elif ast["type"] == "ThisExpression":
        #elif ast["type"] == "FunctionExpression":
        #elif ast["type"] == "NewExpression":
        elif ast["type"] == "UpdateExpression":
            self.property = ast["operator"]
            self.left = BinTree(ast["argument"])
        #elif ObjectExpression
        #elif RegExp
        #elif ArrayExpression
        #elif SequenceExpression
        #else:
        #    print(ast)
        #    print(ast.keys())
        #    print(ast["type"])
        #    print("MISSED")


    def to_list(self, x_lst, edge_lst, token_embedding):
        idx = len(x_lst)
        X = token_embedding[self.type]
        x_lst.append(X)
        if self.left is not None:
            idx_left = self.left.to_list(x_lst, edge_lst, token_embedding)
            edge_lst.append([idx, idx_left])
        if self.right is not None:
            idx_right = self.right.to_list(x_lst, edge_lst, token_embedding)
            edge_lst.append([idx, idx_right])
        return idx

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