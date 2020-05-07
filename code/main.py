import torch

import data
import model
import loss
import option
from trainer.trainer_dbvsr import TRAINER_DBVSR
from trainer.trainer_baseline_lr import TRAINER_BASELINE_LR
from trainer.trainer_baseline_hr import TRAINER_BASELINE_HR
from logger import logger

args = option.args
torch.manual_seed(args.seed)
chkp = logger.Logger(args)

if args.model == 'DBVSR':
    print("Selected model: {}".format(args.model))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = TRAINER_DBVSR(args, loader, model, loss, chkp)
    while not t.terminate():
        t.train()
        t.test()

elif args.model == 'baseline_lr':
    print("Selected model: {}".format(args.model))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = TRAINER_BASELINE_LR(args, loader, model, loss, chkp)
    while not t.terminate():
        t.train()
        t.test()

elif args.model == 'baseline_hr':
    print("Selected model: {}".format(args.model))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = TRAINER_BASELINE_HR(args, loader, model, loss, chkp)
    while not t.terminate():
        t.train()
        t.test()

else:
    raise NotImplementedError('Model [{:s}] is not found'.format(args.model))

chkp.done()
