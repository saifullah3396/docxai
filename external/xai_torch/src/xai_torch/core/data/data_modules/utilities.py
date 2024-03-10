def create_dataset_partitions(dataset, n_partitions, partition_id, seed=0):
    import numpy as np
    from torch.utils.data import Subset

    # get the overall indices
    dataset_indices = list(range(len(dataset)))

    # shuffle the dataset in deterministic way so that every partitioner
    # gets the same shuffled indices
    np.random.seed(seed)
    np.random.shuffle(dataset_indices)

    # generate partitions
    partition = np.array_split(dataset_indices, n_partitions)[partition_id]

    # return the subset with given partition
    return Subset(
        dataset,
        partition,
    )
