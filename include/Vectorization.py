"""Implement functions to apply vectorization for input and to save the files."""

import numpy as np
import json
from tqdm import tqdm


def __load_ent_names(dataset: str, graph_num: int) -> list:
    """Load and return a list of entity names for a given graph.

    :param dataset:   String detailing the dataset. Accepted values: 'fr_en', 'ja_en', 'zh_en' and 'dbp_yg'.
    :param graph_num: Number of the graph of which the entity names are to be loaded.

    :returns:         List containing the entity names of the graph specified.
    """
    # Check the input values for validity:
    if not dataset in ['fr_en', 'ja_en', 'zh_en', 'dbp_yg']:
        raise ValueError('Entered invalid dataset value of "' + dataset + '". Accepted: '
                         + "'fr_en', 'ja_en', 'zh_en' and 'dbp_yg'")
    if not graph_num in [1, 2]:
        ValueError('Entered invalid graph_num value of "' + str(graph_num) + '". Accepted: 1 or 2.')

    # Read the entity names:
    name_list = []
    with open('data/' + dataset + '/ent_ids_' + str(graph_num), encoding='utf-8') as f:
        for line in f:
            current_name = line.split('\t')[-1]
            if '/resource/' in current_name:
                current_name = current_name.split('/resource/')[1]
            if current_name.endswith('\n'):
                name_list.append(current_name[:-1].replace('_', ' '))
            else:
                name_list.append(current_name.replace('_', ' '))
    return name_list


def __load_embedding(embedding_filename="./glove.840B.300d.txt", verbose=True) -> dict:
    """Load a txt file containing a pre-trained embedding-mapping. Return the mapping as dict. \
    Note: Function assumes vector length of 300.

    :param embedding_filename: Name (path) of the .txt file containing the embedding mapping.
    :param verbose:            If true, the loading progress will be displayed.

    :return:                   Embedding dictionary. Keys: the words, values: np.array objects of length 300.
    """
    # Load the embedding file:
    embedding_dict = {}
    with open(embedding_filename, 'r', encoding="utf-8") as embedding_file:
        if verbose:
            print("Loading the embedding file.")
            for line in tqdm(embedding_file):
                values = line.split()
                vec = np.asarray(values[-300:], "float32")
                word = ''.join(values[:-300])
                embedding_dict[word] = vec
        else:
            for line in embedding_file:
                values = line.split()
                vec = np.asarray(values[-300:], "float32")
                word = ''.join(values[:-300])
                embedding_dict[word] = vec
    return embedding_dict


def __embed_entity(ent_name: str, embedding_dict: dict) -> np.array:
    """Derive a vector representation of a string as defined by the embedding dictionary supplied.

    :param ent_name:       Name of the entity.
    :param embedding_dict: Dictionary mapping, key: string, value: corresponding vector representation (list).

    :return:               np.array containing the vector representation of the supplied string.
    """
    if ent_name in embedding_dict.keys():
        return embedding_dict[ent_name]
    elif ' ' in ent_name:
        ent_ls = ent_name.split(' ')
        return np.mean([__embed_entity(el, embedding_dict) for el in ent_ls], axis=0)
    else:
        return np.asarray(np.zeros(300), dtype=np.float32)


def __apply_embedding(dataset: str, embedding_dict: dict, graph_num: int) -> list:
    """Apply the embedding mapping to entity names saved in a .txt file.

    :param dataset:        String detailing the dataset. Accepted values: 'fr_en', 'ja_en', 'zh_en' and 'dbp_yg'.
    :param embedding_dict: Dictonary containing the embedding mapping.
    :param graph_num:      Number of the graph to which the embedding is to be applied.

    :return:               List containing the vectors als lists.
    """
    # Check the input values for validity:
    if dataset not in ['fr_en', 'ja_en', 'zh_en', 'dbp_yg']:
        raise ValueError('Entered invalid dataset value of "' + dataset + '". Accepted: '
                         + "'fr_en', 'ja_en', 'zh_en' and 'dbp_yg'")
    if graph_num not in [1, 2]:
        raise ValueError('Entered invalid graph_number value of "' + str(graph_num) + '". Accepted: 1, 2.')
    # Load the entity names of the KG defined:
    ent_names = __load_ent_names(dataset=dataset, graph_num=graph_num)
    # Apply embedding to the KG:
    vector_list = []
    for ent_name in ent_names:
        vector_list.append(__embed_entity(ent_name, embedding_dict))
    return vector_list


def __store_embedding(dataset: str, embedding_dict: dict):
    """Apply embedding to the given dataset and store it as a .json file. \
    Note: This will overwrite existing files!

    :param dataset:        String detailing the dataset for which to derive the embedding. \
                           Accepted values: 'fr_en', 'ja_en', 'zh_en' and 'dbp_yg'.
    :param embedding_dict: Dictonary containing the embedding mapping.
    """
    ent_ls_1 = __apply_embedding(dataset=dataset, embedding_dict=embedding_dict, graph_num=1)
    ent_ls_2 = __apply_embedding(dataset=dataset, embedding_dict=embedding_dict, graph_num=2)
    vec_ls = ent_ls_2 + ent_ls_1
    vec_ls = [vec.tolist() for vec in vec_ls]
    with open(file='data/' + dataset + '/' + dataset.split('_', 1)[0] + '_vectorList.json',
              mode='w',
              encoding='utf-8') as outfile:
        json.dump(vec_ls, outfile)


if __name__ == '__main__':
    embedding_dict = __load_embedding()
    # Note: The following function call may overwrite an existing file!
    __store_embedding('dbp_yg', embedding_dict=embedding_dict)
