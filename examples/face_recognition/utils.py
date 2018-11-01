# @File  : utils.py
# @Author: X.Yang
# @Contact : pistonyang@gmail.com
# @Date  : 18-11-1


def inf_train_gen(loader):
    """
    Using iterations train network.
    :param loader: Dataloader
    :return: batch of data
    """
    while True:
        for batch in loader:
            yield batch