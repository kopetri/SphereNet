import pytorch_lightning as pl
from network import SphereNet
import torch
import numpy as np

class ShpereNetModule(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.model = SphereNet()
        self.criterion = torch.nn.MSELoss()

    def forward(self, batch, batch_idx, name):
        X, Y = batch
        Y_hat, _ = self.model(X)
        assert not torch.any(torch.isnan(X)), "X has NaN values!"
        assert not torch.any(torch.isnan(Y)), "GT has NaN values!"
        assert not torch.any(torch.isnan(Y_hat)), "prediction has NaN values!"
        loss = self.criterion(Y_hat, Y)
        assert not torch.any(torch.isnan(loss)), "Loss has NaN values!"
        self.log('{}_loss'.format(name), loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self(batch, batch_idx, "test")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(self.opt.learning_rate_decay, self.global_step)),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        return [optimizer], [scheduler]