def print_learning_rate(optimizer):
    """
    打印优化器中每个参数组的当前学习率。

    :param optimizer: PyTorch 的优化器实例
    """
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Learning rate for param group {i}: {param_group['lr']}")