import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam


class LightningNet(LightningModule):
    def __init__(self, options, data=None):
        super(LightningNet, self).__init__()
        self.options = options
        self.data = data
        self.layers = nn.ModuleList()

        n_layers = options['n_layers']
        dropout = options['dropout']
        input_dim = options['n_input']
        n_classes = options['n_classes']

        # Add one linear and one dropout layer per layer
        for i in range(n_layers):
            output_dim = options['n_units_l{}'.format(i)]
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.layers.append(nn.Linear(input_dim, n_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if type(layer) != nn.Dropout:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)

    def train_dataloader(self):
        X = torch.tensor(self.data['X_train'], dtype=torch.float)
        y = torch.tensor(self.data['y_train'], dtype=torch.long)

        return DataLoader(TensorDataset(X, y),
                          batch_size=self.options['batch_size'],
                          shuffle=False)

    def val_dataloader(self):
        X = torch.tensor(self.data['X_valid'], dtype=torch.float)
        y = torch.tensor(self.data['y_valid'], dtype=torch.long)

        return DataLoader(TensorDataset(X, y),
                          batch_size=self.options['batch_size'],
                          shuffle=False)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.nll_loss(outputs, labels)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.nll_loss(outputs, labels)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        return Adam(self.parameters(),
                    lr=self.options['learning_rate'])
