Efficient DL: Multi-channel images

# Setup

1/ Install Morphem Evaluation Benchmark package: 

https://github.com/broadinstitute/MorphEm


2/ Install required packages:

`pip install -r requirements.txt`

or

`conda env create -f environment.yml`


# Add Wandb key
If you want to use Wandb to keep track of experiments, add your Wandb key to `.env` file:

`echo WANDB_API_KEY=your_wandb_key >> .env`

Otherwise, change `use_wandb` to False in configs/morphem70k/logging/wandb.yaml to disable Wandb.

# Dataset
The dataset can be found at https://doi.org/10.5281/zenodo.7988357


# Training

In this project, we use [Hydra](https://hydra.cc/) to manage configurations.
To submit a job, you need to specify the config file:

```
-m: multi-run mode

-cp: config folder, all config files are in `configs/morphem70k`

-cn: config file name (without the .yaml extension)
```

Parameters in the command lines will override the ones in the config file.
Here is an example to submit a job to train a SliceParam model:

```
python main.py -m -cp configs/morphem70k -cn morphem70k_cfg model=sliceparam tag=slice ++optimizer.params.lr=0.00001 ++model.first_layer=pretrained_pad_dups ++model.learnable_temp=True ++model.temperature=0.07
```


