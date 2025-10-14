from torch.utils.data import DataLoader


class GTSRBDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
