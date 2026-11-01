"""Train the total hidden-correction AMG-GAT model."""

import os
import runpy


if __name__ == '__main__':
    os.environ['MODEL'] = 'total'
    runpy.run_module('training.train_amggat', run_name='__main__')
