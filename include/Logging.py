"""Provide logging functionality."""

import os
import pandas as pd


def fresh_log_file(dataset: str, mode: str, time_str: str, loss_ls: list=None) -> str:
    """Create an empty log file to log the training details. Save the file in folder 'logging'. \
    Limitation: Measures are assumed to be Hits@1, Hits@10, Hits@50 and Hits@100.

    :param dataset: String indicating the dataset. Accepted: "'ja_en', 'zh_en', 'ja_en', 'fr_en', 'dpb_yg'."

    :return:        Training file name.
    """
    # Assert the validity of the parameters:
    if not dataset in ['fr_en', 'ja_en', 'zh_en', 'dbp_yg']:
        raise ValueError('Entered invalid dataset value of "' + dataset + '". Accepted: '
                         + "'fr_en', 'ja_en', 'zh_en' and 'dbp_yg'")
    if not mode in ['training', 'result', 'losses']:
        raise ValueError('Entered invalid mode value of "' + mode + '". Accepted: '
                         + "'training', 'result' and 'losses'")
    # Create a fresh log file:
    if mode in ['training', 'result']:
        log_filename = dataset + '_' + mode + '_' + time_str + '.csv'
        pd.DataFrame({'left_Hits@1': [],
                      'left_Hits@10': [],
                      'left_Hits@50': [],
                      'left_Hits@100': [],
                      'right_Hits@1': [],
                      'right_Hits@10': [],
                      'right_Hits@50': [],
                      'right_Hits@100': [],
                      'loss': []}).to_csv(
            os.path.join('./logging/', log_filename), sep='|', index=False)
    elif mode == 'losses':
        log_filename = dataset + '_losses_' + time_str + '.txt'
        with open(os.path.join('logging/', log_filename), 'w') as f:
            for item in loss_ls:
                f.write("%s\n" % item)
    return log_filename
