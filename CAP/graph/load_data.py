import random
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Flickr, TUDataset

try:
    import torch_geometric.io.fs as _fs
    _orig_mv = _fs.mv
    def _mv_safe(path1, path2, recursive=False):
        try:
            return _orig_mv(path1, path2, recursive)
        except TypeError:
            import shutil, os
            os.makedirs(os.path.dirname(path2), exist_ok=True)
            return shutil.move(path1, path2)
        except Exception:
            import shutil, os
            os.makedirs(os.path.dirname(path2), exist_ok=True)
            return shutil.move(path1, path2)
    _fs.mv = _mv_safe
except Exception:
    pass

def load_graph_data(dataset_name, data_folder):
    if dataset_name in ['ENZYMES', 'NCI109', 'MUTAG', 'PROTEINS']:
        dataset = TUDataset(f'{data_folder}/TUDataset/', name=dataset_name)
    else:
        raise ValueError('Error: invalid dataset name! Supported datasets: [ENZYMES, NCI109, MUTAG, PROTEINS]')

    input_dim = dataset.num_features
    output_dim = dataset.num_classes
    return dataset, input_dim, output_dim


def GraphDownstream(data, shots=5, test_fraction=0.4):
    num_classes = data.y.max().item() + 1
    graph_list = []
    for c in range(num_classes):
        indices = torch.where(data.y.squeeze() == c)[0].tolist()
        if len(indices) < shots:
            graph_list.extend(indices)
        else:
            graph_list.extend(random.sample(indices, k=shots))
    random_graph_list = random.sample(range(len(data)), k=len(data))
    for graph in graph_list:
        random_graph_list.remove(graph)
    train_data = [data[g] for g in graph_list]
    test_data = [data[g] for g in random_graph_list[:int(test_fraction * len(data))]]

    return train_data, test_data
