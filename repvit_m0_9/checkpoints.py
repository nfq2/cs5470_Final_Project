"""Keep separate validation-selected and latest completed-epoch models."""
from pathlib import Path

import torch


class CheckpointWriter:
    def __init__(self, output):
        self.output = Path(output)
        self.best_loss = float('inf')
        self.best_balanced = float('-inf')
        self.balanced_loss = float('inf')

    def _write(self, name, checkpoint, criterion):
        # Replace only after the new file is fully written.
        temporary = self.output / (name + '.tmp')
        torch.save(dict(checkpoint, selection_criterion=criterion), temporary)
        temporary.replace(self.output / name)

    def save(self, checkpoint):
        valid = checkpoint['validation']
        loss, balanced = valid['loss'], valid['balanced_accuracy']
        if loss < self.best_loss:
            self._write('best.pt', checkpoint, 'minimum unweighted validation loss')
            self.best_loss = loss
        if balanced is not None and (balanced > self.best_balanced or
                                     (balanced == self.best_balanced and loss < self.balanced_loss)):
            self._write('best_balanced.pt', checkpoint,
                        'maximum validation balanced accuracy; ties use lower validation loss')
            self.best_balanced, self.balanced_loss = balanced, loss
        self._write('last.pt', checkpoint, 'latest completed epoch; not validation-selected')
