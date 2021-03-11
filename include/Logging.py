"""Provide logging functionality."""

import os
from datetime import datetime
import pandas as pd


def fresh_training_log() -> str:
    """Create an empty log file to log the training details. Save the file in folder 'logging'. \
    Limitation: Measures are assumed to be Hits@1, Hits@10, Hits@50 and Hits@100.

    :return: Training file name.
    """
    now_string = datetime.strftime(datetime.now(tz=None), "%Y_%m_%d_%H.%M.%S")
    train_log_filename = 'training_log_' + now_string + '.csv'
    pd.DataFrame({'left_Hits@1': [],
                  'left_Hits@10': [],
                  'left_Hits@50': [],
                  'left_Hits@100': [],
                  'right_Hits@1': [],
                  'right_Hits@10': [],
                  'right_Hits@50': [],
                  'right_Hits@100': [],
                  'loss': []}).to_csv(
        os.path.join('./logging/', train_log_filename), sep='|', index=False)

    return train_log_filename
