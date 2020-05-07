import decimal
import torch
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
import torch.optim as optim


class TRAINER_BASELINE_LR(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(TRAINER_BASELINE_LR, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using TRAINER_BASELINE_LR")
        self.optimizer = self.make_optimizer()

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        a = optim.Adam([{"params": self.model.model.pwcnet.parameters(), "lr": self.args.pwc_lr},
                        {"params": self.model.model.sr_net.parameters(), "lr": self.args.lr}], **kwargs)
        return a

    def train(self):
        print("Now training")

        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()
        self.ckp.write_log('Epoch {:3d} with pwcLr {:.2e}\t rcanLr {:.2e}'.format(epoch, decimal.Decimal(lr[0]),
                                                                                  decimal.Decimal(lr[1])))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()

        for batch, (input, gt, _, _, _) in enumerate(self.loader_train):

            if self.args.n_colors == 1 and input.size()[-3] == 3:
                raise Exception("Now just Support RGB mode, not Support Ycbcr mode! "
                                "See args.n_colors={} and input_channel={}"
                                .format(self.args.n_colors, input.size()[-3]))

            input = input.to(self.device)
            gt = gt[:, self.args.n_sequences // 2, :, :, :].to(self.device)

            sr_output = self.model(input)

            self.optimizer.zero_grad()
            loss = self.loss(sr_output, gt)
            loss.backward()
            self.optimizer.step()
            self.ckp.report_log(loss.item())

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch)
                ))

        self.loss.end_log(len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input, gt, filename, kernel, _) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequences // 2][0]

                if self.args.n_colors == 1 and input.size()[-3] == 3:
                    raise Exception("Now just Support RGB mode, not Support Ycbcr mode! "
                                    "See args.n_colors={} and input_channel={}"
                                    .format(self.args.n_colors, input.size()[-3]))

                input = input.to(self.device)
                gt = gt[:, self.args.n_sequences // 2, :, :, :].to(self.device)

                sr_output = self.model(input)

                PSNR = utils.calc_psnr(gt, sr_output, rgb_range=self.args.rgb_range,
                                       is_rgb=True)

                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:
                    sr_output = utils.postprocess(sr_output, rgb_range=self.args.rgb_range, ycbcr_flag=False,
                                                  device=self.device)[0]
                    save_list = [sr_output]
                    self.ckp.save_images(filename, save_list, self.args.testset)

        self.ckp.end_log(len(self.loader_test), train=False)
        best = self.ckp.psnr_log.max(0)
        self.ckp.write_log('[{}]\t{} average PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
            self.args.data_test, self.args.testset, self.ckp.psnr_log[-1],
            best[0], best[1] + 1))
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
