# Baseline experiment on the implemented Abstract CVRP

Preparing conda environment

```bash
# conda deactivate && conda env remove -n attention

# install essential dependencies and dev environment
conda create -n attention "python<3.10" pip setuptools numpy scipy matplotlib scikit-learn \
    "pytorch::pytorch" notebook \
  && conda activate attention \
  && pip install einops tqdm black pre-commit gitpython \
  && pre-commit install
```

**Note** that we use `--no_tensorboard` flag to avoid TB logging, since this would require extra PyPi dependency `pip install tensorboard_logger`.

Train the edge-weighted graph attention model on abstract CVRP with 50 nodes, and then the baseline model.

```bash
# train the abscvrp-50 model
python run.py --problem abscvrp --graph_size 50 --model attention --no_tensorboard --baseline rollout --run_name abscvrp

# train the baseline cvrp-50 model
python run.py --problem abscvrp --graph_size 50 --model attention --no_tensorboard --baseline rollout --run_name cvrp

# produce vrp-50 data
python generate_data.py --problem vrp --name test --seed 1234
```

Test the trained edge and baseline models.

```bash
# evaluating the vrp-50 model on vrp-50 dataset: greedy
python eval.py data/vrp/vrp50_test_seed1234.pkl -f --model outputs/cvrp_50/vrp_20220626T095922/ --decode_strategy greedy
# Average cost: 11.022562026977539 +- 0.0261368465423584
# Average serial duration: 0.17068865356445312 +- 0.002893510008497786
# Average parallel duration: 0.00016668813824653625
# Calculated total duration: 0:00:01

# evaluating the abscvrp-50 model on vrp-50 dataset
python eval.py data/vrp/vrp50_test_seed1234.pkl -f --model outputs/abscvrp_50/abscvrp_20220722T134643/ --decode_strategy greedy
# Average cost: 38.86709213256836 +- 0.16148748397827148
# Average serial duration: 0.33082456283569334 +- 0.0028162842847413933
# Average parallel duration: 0.0003230708621442318
# Calculated total duration: 0:00:03

# evaluating the vrp-50 model on vrp-20 dataset: greedy
python eval.py data/vrp/vrp20_test_seed1234.pkl -f --model outputs/cvrp_50/vrp_20220626T095922/ --decode_strategy greedy
# Average cost: 6.662790775299072 +- 0.01814401865005493
# Average serial duration: 0.09944922180175782 +- 0.002876180305271037
# Average parallel duration: 9.711838066577912e-05
# Calculated total duration: 0:00:00

# evaluating the abscvrp-50 model on vrp-20 dataset
python eval.py data/vrp/vrp20_test_seed1234.pkl -f --model outputs/abscvrp_50/abscvrp_20220722T134643/ --decode_strategy greedy
# Average cost: 12.036279678344727 +- 0.046112346649169925
# Average serial duration: 0.1292957202911377 +- 0.002762338773087356
# Average parallel duration: 0.00012626535184681416
# Calculated total duration: 0:00:01
```
