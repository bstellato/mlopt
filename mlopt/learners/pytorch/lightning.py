import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import TensorDataset, DataLoader
import mlopt.learners.pytorch.settings as stg
from torch.optim import Adam


class Net(nn.Module):
    def __init__(self, bounds, n_input, n_classes, trial):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.bounds = bounds

        n_layers = trial.suggest_int(
            'n_layers', *self.bounds['n_layers'])
        dropout = trial.suggest_uniform('dropout', *stg.PARAMETER_BOUNDS)
        input_dim = n_input

        # Add one linear and one dropout layer per layer
        for i in range(n_layers):
            output_dim = trial.suggest_int(
                'n_units_l{}'.format(i), *self.bounds['n_units_l'],
                log=True)
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.layers.append(nn.Dropout(dropout))

            input_dim = output_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if type(layer) != nn.Dropout:
                x = F.relu(x)
        return F.log_softmax(x)


class LightningNet(LightningModule):
    def __init__(self, data, bounds, n_input, n_classes, trial):
        super(LightningNet, self).__init__()
        self.model = Net(bounds, n_input, n_classes, trial)
        self.bounds = bounds

        # Suggest parameters
        self.batch_size = trial.suggest_int(
            'batch_size', *self.bounds['batch_size'])
        self.learning_rate = trial.suggest_float(
            'learning_rate', *self.bounds['learning_rate'], log=True)

        # Split data in train and validation
        self.data = data

    def forward(self, data):
        return self.model(data)

    def train_dataloader(self):
        X = self.torch.tensor(self.data['X_train'], dtype=self.torch.float)
        y = self.torch.tensor(self.data['y_train'], dtype=self.torch.long)

        return DataLoader(TensorDataset(X, y), batch_size=self.batch_size,
                          shuffle=False)

    def val_dataloader(self):
        X = self.torch.tensor(self.data['X_valid'], dtype=self.torch.float)
        y = self.torch.tensor(self.data['y_valid'], dtype=self.torch.long)

        return DataLoader(TensorDataset(X, y), batch_size=self.batch_size,
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
        return Adam(self.model.parameters(), lr=self.learning_rate)
