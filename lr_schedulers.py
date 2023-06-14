from timm.scheduler import MultiStepLRScheduler, CosineLRScheduler, PlateauLRScheduler


def create_my_scheduler(optimizer, scheduler_type: str,  config: dict):
    if scheduler_type == 'multistep':
        scheduler = MultiStepLRScheduler(optimizer, **config)
    elif scheduler_type == 'cosine':
        scheduler = CosineLRScheduler(optimizer, **config)
    # elif scheduler_type == 'plateau': TODO: add metric
    #     scheduler = PlateauLRScheduler(optimizer, **config)
    else:
        raise NotImplementedError(f'Not implemented scheduler: {scheduler_type}')
    return scheduler



