import os
import logging
import torch
from pathlib import Path
import fasttext

from model import Net
import utils
from run_bug_finding import read_json_file



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
            data_dict[json_file] = utils.generate_data_dict(if_ast, token_embedding, y)
            #print(data.is_undirected())

    if len_initial < len(data_dict):
        logging.info("Save new dict")
        torch.save(data_dict, serialize_file)

    logging.info("Generate data list")
    data_lst = []
    for data_key, data_value in data_dict.items():
        data_lst.append(data_value)

    data_lst = data_lst[:50000]
    #print("Ratio: ", str(len(data_lst)/1000))

    ### Load model
    logging.info("Load model")
    net = utils.load_model(model_path)
    logging.info(net)

    ### Train
    net.train(data_lst, 1e-3, 100)


if __name__ == '__main__':
    run()




