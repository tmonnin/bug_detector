import os
import logging
from collections import defaultdict
import torch
from pathlib import Path
import fasttext
from model import Net
import utils



def run():
    torch.multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)

    model_path = "model"
    data_path = "../../json_files_augmented_new"
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
        data_dict = defaultdict(list)

    logging.info("Read files")
    list_of_json_file_paths = list(Path(data_path).glob('**/*.json'))
    list_of_json_file_paths = [str(p) for p in list_of_json_file_paths]
    len_initial = len(data_dict)
    for index, json_file in enumerate(list_of_json_file_paths):
        if not json_file in data_dict.keys():
            logging.info(str(index) + "/" + str(len(list_of_json_file_paths)) + " - " + json_file)
            d_lst = utils.read_json_file(json_file)
            for d in d_lst:
                # dict keys: if_ast, condition, code_adjacent, label
                data_dict[json_file].append(utils.generate_data_dict_sequence(d, token_embedding))

    if len_initial < len(data_dict):
        logging.info("Save new dict")
        torch.save(data_dict, serialize_file)

    logging.info("Generate data list")
    data_lst = []
    for data_key, data_file_lst in data_dict.items():
        data_lst += data_file_lst
    del data_dict # Free resources to improve debugging performance

    data_lst = data_lst[:120000]
    print("Training data length: ", str(len(data_lst)))
    labels = []
    [labels.append(data["label"]) for data in data_lst]
    distribution = [0.8, 0.05, 0.05, 0.0, 0.05, 0.05]
    logging.info("Generate weight list")
    weights = utils.weighted_distribution(labels, distribution)

    ### Load model
    logging.info("Load model")
    net = utils.load_model(model_path)
    logging.info(net)

    ### Train
    net.train(data_lst, 1e-3, 100, weights)


if __name__ == '__main__':
    run()




