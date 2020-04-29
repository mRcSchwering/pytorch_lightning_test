# Pytorch Lightning Test

I'm trying to find a good pattern for using [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning).
Created some dummy tasks here.
Things to consider:

1. Need to tune by searching hyperparameter space.
2. Have more than 1 training run per trial (hyperparameter set). _E.g. fold1, fold2_.
3. Log summaries with (at least) best val loss achieved and hyperparameter set for all trials.
4. Log epoch-wise training progress (at least train/val loss).
5. Have multiple training stages (_e.g._ warmup rounds).
6. Possibility to analyze gradients.
7. Run in distributed environment.
8. Evaluate dataloading on every step.
