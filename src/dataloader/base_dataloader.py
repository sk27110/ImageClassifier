from torch.utils.data import DataLoader


class GTSRBDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        cfg 
    ):
        super().__init__(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=cfg.dataloader.shuffle,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory
        )



# def get_dataloader(dataset, cfg):
#     dataloader = DataLoader(
#         dataset,
#         batch_size=cfg.dataloader.batch_size,
#         shuffle=cfg.dataloader.shuffle,
#         num_workers=cfg.dataloader.num_workers,
#         pin_memory=cfg.dataloader.pin_memory
#     )
#     return dataloader