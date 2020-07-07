import os
import logging
import torch
from pathlib import Path
import fasttext

from model import Net
import utils
from run_bug_finding import read_json_file

from torch_geometric.data import Data


def run():
    torch.multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)

    model_path = "model"
    data_path = "../../json_files_augmented"
    embedding_path = "../../json_files/all_token_embedding.bin"
    serialize_file = "../../data_full.pt"

    ### Load data
    logging.info("Load embedding")
    token_embedding = fasttext.load_model(path=embedding_path)

    logging.info("Read dict")
    try:
        data_dict = torch.load(serialize_file)
    except FileNotFoundError:
        logging.warning("No dict found, create new")
        data_dict = {}

    logging.info("Read files")
    list_of_json_file_paths = list(Path(data_path).glob('**/*.json'))
    list_of_json_file_paths = [str(p) for p in list_of_json_file_paths]
    len_initial = len(data_dict)
    for index, json_file in enumerate(list_of_json_file_paths[0:50000]):
        if not json_file in data_dict.keys():
            logging.info(str(index) + "/" + str(len(list_of_json_file_paths)) + " - " + json_file)
            if_ast, y = read_json_file(json_file)
            conditional_handler = utils.ConditionalHandler(None, None, if_ast)
            x_lst = []
            edge_lst = []
            #conditional_handler.bin_tree.to_list(x_lst, edge_lst, token_embedding)
            conditional_handler.bin_tree.to_flattened(x_lst, edge_lst, token_embedding, depth=4)
            type_oh = torch.zeros([15], dtype=torch.int64) # [b, c, h, w]
            property_ft = torch.zeros((100,1,15), dtype=torch.float) # [b, c, h, w]

            for i in range(len(x_lst)):
                type_oh[i] = x_lst[i][0]
                #type_oh[:,0,i] = x_lst[i][0]
                property_ft[:, 0, i] = x_lst[i][1]

            #x = torch.tensor(x_lst, dtype=torch.float)
            edge_index = None# torch.tensor(edge_lst, dtype=torch.long)

            #if len(x) < 5:
            #    continue
            #x=x.unsqueeze(0).unsqueeze(0)
            data = Data(x=type_oh, edge_index=edge_index)
            label = torch.tensor([y], dtype=torch.float32)
            data_dict[json_file] = {'type_oh': type_oh, 'property_ft': property_ft, 'data': data, 'label': label}
            #print(data.is_undirected())

    if len_initial < len(data_dict):
        logging.info("Save new dict")
        torch.save(data_dict, serialize_file)

    logging.info("Generate data list")
    data_lst = []
    for data_key, data_value in data_dict.items():
        data_lst.append(data_value)

    data_lst = data_lst[:10000]
    #print("Ratio: ", str(len(data_lst)/1000))


    ### Load model
    logging.info("Load model")
    net = Net()
    logging.info(net)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

    ### Train
    net.train(data_lst, 1e-3, 100)


if __name__ == '__main__':
    run()




