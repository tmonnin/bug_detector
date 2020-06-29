import os
import json
import numpy as np
import torch
from pathlib import Path
import fasttext

from model import Net
import utils

from torch_geometric.data import Data


def run():
    torch.multiprocessing.freeze_support()

    model_path = "model"
    data_path = "../../json_files_augmented"
    embedding_path = "../../json_files/all_token_embedding.bin"

    ### Load data
    token_embedding = fasttext.load_model(
        path=embedding_path)
    list_of_json_file_paths = list(Path(data_path).glob('**/*.json'))
    list_of_json_file_paths = [str(p) for p in list_of_json_file_paths]
    data_lst = []
    for json_file in list_of_json_file_paths[0:1000]:
        with open(json_file) as file:
            if_ast, y = json.load(file)
            conditional_handler = utils.ConditionalHandler(None, None, if_ast)
            x_lst = []
            edge_lst = []
            conditional_handler.bin_tree.to_list(x_lst, edge_lst, token_embedding)
            x = torch.tensor(x_lst, dtype=torch.float)
            edge_index = torch.tensor(edge_lst, dtype=torch.long)
            if len(x) < 5:
                continue
            data = Data(x=x, edge_index=edge_index)
            data_lst.append([data, torch.tensor([y], dtype=torch.int)])
            #print(data.is_undirected())
    print("Ratio: ", str(len(data_lst)/1000))
    ### Load model
    net = Net()
    print(net)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

    ### Train
    net.train(data_lst, 1e-3, 10)
    torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    run()




