## Allen
python main.py -m -cp configs/morphem70k -cn allen_cfg optimizer=adamw tag=allen ++optimizer.params.lr=0.001 ++model.unfreeze_first_layer=True ++model.unfreeze_last_n_layers=-1 ++model.first_layer=pretrained_pad_avg ++model.temperature=0.3 ++train.seed=582814

## HPA
python main.py -m -cp configs/morphem70k -cn hpa_cfg optimizer=adamw tag=hpa ++optimizer.params.lr=0.0001 ++model.unfreeze_first_layer=True ++model.unfreeze_last_n_layers=-1 ++model.first_layer=pretrained_pad_avg ++model.temperature=0.3 ++train.seed=744395

## CP
python main.py -m -cp configs/morphem70k -cn cp_cfg optimizer=adamw tag=cp ++optimizer.params.lr=0.0001 ++model.unfreeze_first_layer=True ++model.unfreeze_last_n_layers=-1 ++model.first_layer=pretrained_pad_avg ++model.temperature=0.3 ++train.seed=530400

## Depthwise
python main.py -m -cp configs/morphem70k -cn morphem70k_cfg model=depthwiseconvnext tag=depthwise ++optimizer.params.lr=0.0004 ++model.kernels_per_channel=64 ++model.pooling_channel_type=weighted_sum_random ++model.temperature=0.07 ++train.seed=483112

## TargetParam (Shared)
python main.py -m -cp configs/morphem70k -cn morphem70k_cfg model=separate tag=shared ++optimizer.params.lr=0.0002 ++model.learnable_temp=True ++model.temperature=0.07 ++train.seed=505429

## SliceParam
python main.py -m -cp configs/morphem70k -cn morphem70k_cfg model=sliceparam tag=slice ++optimizer.params.lr=0.0001 ++model.learnable_temp=True ++model.temperature=0.15 ++model.first_layer=pretrained_pad_dups ++model.slice_class_emb=True ++train.seed=725375

## HyperNet
python main.py -m -cp configs/morphem70k -cn morphem70k_cfg model=hyperconvnext tag=hyper ++optimizer.params.lr=0.0004 ++model.separate_emb=True ++model.temperature=0.07 ++model.z_dim=128 ++model.hidden_dim=256 ++train.seed=125617

## Template Mixing
python main.py -m -cp configs/morphem70k -cn morphem70k_cfg model=template_mixing tag=templ ++optimizer.params.lr=0.0001 ++model.num_templates=128 ++model.temperature=0.05 ++model.separate_coef=True ++train.seed=451006
