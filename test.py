from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# TODO: use SummaryWriter directly
# only change the way directories are named (or not? actually doesnt matter)
# write logger from that...

#this_dir = Path(__file__).parent.absolute()
this_dir = Path('e5_using_logkey').absolute()

# TODO: write logging class with this...
with SummaryWriter(log_dir=this_dir / 'logs') as w:
    for i in range(5):
        w.add_scalar('y=2x', i * 2, i)
    w.add_hparams({'lr': 0.1*i, 'bsize': i}, {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})


event_acc = EventAccumulator(str(this_dir / 'logs' / '0' / 'version_0'))
event_acc.Reload()
print(event_acc.Tags())


event_acc.scalars.Items('best/val-loss')[-1]
event_acc.scalars.Items('best/epoch')[-1]







# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('hparam/accuracy'))

vals