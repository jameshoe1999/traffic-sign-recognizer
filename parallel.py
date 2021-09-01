from multiprocessing import cpu_count, Pool
from multiprocessing.pool import AsyncResult
import common as cm
import numpy as np

def flatten(list2D: list[list]) -> list[any]:
    return [item for sublist in list2D for item in sublist]

def preprocess_image_parallel(dataset: np.ndarray, labels: list[str]):
    pool_num = min(cpu_count(), 10)
    datasize = len(dataset)
    load_per_core = datasize // pool_num
    print(f'Data size: {datasize} and data chunk per core: {load_per_core}')
    results: list[AsyncResult] = []
    with Pool(processes=pool_num) as pool:
        X = []
        Y = []
        for x in range(pool_num):
            start = x * load_per_core
            end = start + load_per_core
            data_subset = dataset[start:end]
            label_subset = labels[start:end]
            task = pool.apply_async(preprocess_image_child, (data_subset, label_subset))
            results.append(task)
        for result in results:
            x, y = result.get()
            X.append(x)
            Y.append(y)
        X = np.array(flatten(X))
        Y = flatten(Y)
        return X, Y

def preprocess_image_child(dataset: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    X = [] # dataset
    Y = [] # label
    for (index, image) in enumerate(dataset):
        result = cm.preprocess_image(image)
        if (result is None):
            continue
        X.append(result)
        Y.append(labels[index])
    X = np.array(X)
    return X, Y