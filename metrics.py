"""
多目标优化性能评价指标

包含:
- GD (Generational Distance): 代距离，衡量收敛性
- SP (Spacing): 间距，衡量分布均匀性
"""

import numpy as np
from typing import List, Union


def generational_distance(obtained_front: Union[List, np.ndarray],
                          true_front: Union[List, np.ndarray]) -> float:
    """
    计算代距离 (Generational Distance, GD)

    GD衡量算法得到的Pareto前沿与真实Pareto前沿的逼近程度。
    GD越小，说明算法得到的解集越逼近真实解集。

    公式: GD = (1/n) * sqrt(sum(di^2))
    其中 di 是第i个解到真实Pareto前沿的最小欧氏距离

    Parameters:
    -----------
    obtained_front : Union[List, np.ndarray]
        算法得到的Pareto前沿 (n_solutions, n_objectives)
    true_front : Union[List, np.ndarray]
        真实的Pareto前沿 (n_true_solutions, n_objectives)

    Returns:
    --------
    gd : float
        代距离值
    """
    # 转换为numpy数组
    obtained_front = np.array(obtained_front)
    true_front = np.array(true_front)

    if len(obtained_front) == 0:
        return float('inf')

    if len(true_front) == 0:
        raise ValueError("真实Pareto前沿不能为空")

    n = len(obtained_front)
    distances = []

    # 对每个获得的解，计算到真实前沿的最小距离
    for point in obtained_front:
        # 计算到所有真实前沿点的距离
        dists = np.sqrt(np.sum((true_front - point) ** 2, axis=1))
        # 取最小距离
        min_dist = np.min(dists)
        distances.append(min_dist)

    # 计算GD
    distances = np.array(distances)
    gd = np.sqrt(np.sum(distances ** 2)) / n

    return gd


def spacing(obtained_front: Union[List, np.ndarray]) -> float:
    """
    计算间距 (Spacing, SP)

    SP衡量算法得到的Pareto前沿的均匀性。
    SP越小，说明解集分布越均匀。

    公式: SP = sqrt((1/(n-1)) * sum((d_bar - di)^2))
    其中:
    - di = min_j(sum_k(|f_k^i - f_k^j|))，i和j是不同的解
    - d_bar 是所有di的平均值

    Parameters:
    -----------
    obtained_front : Union[List, np.ndarray]
        算法得到的Pareto前沿 (n_solutions, n_objectives)

    Returns:
    --------
    sp : float
        间距值
    """
    # 转换为numpy数组
    obtained_front = np.array(obtained_front)

    n = len(obtained_front)

    if n < 2:
        return 0.0

    distances = []

    # 对每个解，计算到其他解的最小距离
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i == j:
                continue
            # 计算曼哈顿距离
            dist = np.sum(np.abs(obtained_front[i] - obtained_front[j]))
            min_dist = min(min_dist, dist)
        distances.append(min_dist)

    distances = np.array(distances)
    d_bar = np.mean(distances)

    # 计算SP
    sp = np.sqrt(np.sum((d_bar - distances) ** 2) / (n - 1))

    return sp


def hypervolume(obtained_front: Union[List, np.ndarray],
                reference_point: Union[List, np.ndarray]) -> float:
    """
    计算超体积 (Hypervolume, HV)

    HV衡量Pareto前沿覆盖的目标空间体积。
    HV越大，说明解集质量越好。

    注意：这是一个简化版本，仅适用于2维目标

    Parameters:
    -----------
    obtained_front : Union[List, np.ndarray]
        算法得到的Pareto前沿 (n_solutions, 2)
    reference_point : Union[List, np.ndarray]
        参考点（通常是最坏点）

    Returns:
    --------
    hv : float
        超体积值
    """
    obtained_front = np.array(obtained_front)
    reference_point = np.array(reference_point)

    if len(obtained_front) == 0:
        return 0.0

    if obtained_front.shape[1] != 2:
        raise NotImplementedError("当前仅支持2维目标的超体积计算")

    # 按第一个目标排序
    sorted_front = obtained_front[obtained_front[:, 0].argsort()]

    # 计算超体积
    hv = 0.0
    prev_x = reference_point[0]

    for point in sorted_front:
        if point[0] >= reference_point[0] or point[1] >= reference_point[1]:
            continue
        width = prev_x - point[0]
        height = reference_point[1] - point[1]
        hv += width * height
        prev_x = point[0]

    return hv


def inverted_generational_distance(obtained_front: Union[List, np.ndarray],
                                   true_front: Union[List, np.ndarray]) -> float:
    """
    计算反向代距离 (Inverted Generational Distance, IGD)

    IGD衡量真实Pareto前沿到算法得到的前沿的距离。
    IGD越小越好，同时考虑了收敛性和分布性。

    Parameters:
    -----------
    obtained_front : Union[List, np.ndarray]
        算法得到的Pareto前沿
    true_front : Union[List, np.ndarray]
        真实的Pareto前沿

    Returns:
    --------
    igd : float
        反向代距离值
    """
    # IGD是从真实前沿到获得前沿的距离
    return generational_distance(true_front, obtained_front)


def evaluate_performance(obtained_front: Union[List, np.ndarray],
                         true_front: Union[List, np.ndarray],
                         reference_point: Union[List, np.ndarray] = None) -> dict:
    """
    综合评估算法性能

    Parameters:
    -----------
    obtained_front : Union[List, np.ndarray]
        算法得到的Pareto前沿
    true_front : Union[List, np.ndarray]
        真实的Pareto前沿
    reference_point : Union[List, np.ndarray], optional
        用于计算超体积的参考点

    Returns:
    --------
    metrics : dict
        包含各项指标的字典
    """
    metrics = {}

    # 计算GD
    metrics['GD'] = generational_distance(obtained_front, true_front)

    # 计算SP
    metrics['SP'] = spacing(obtained_front)

    # 计算IGD
    metrics['IGD'] = inverted_generational_distance(obtained_front, true_front)

    # 计算超体积（如果提供了参考点且是2维）
    if reference_point is not None:
        obtained_front_array = np.array(obtained_front)
        if obtained_front_array.shape[1] == 2:
            try:
                metrics['HV'] = hypervolume(obtained_front, reference_point)
            except:
                pass

    return metrics


if __name__ == "__main__":
    # 测试指标计算
    print("测试性能评价指标")
    print("-" * 50)

    # 创建测试数据
    # 真实Pareto前沿
    true_front = np.array([
        [0.0, 1.0],
        [0.25, 0.75],
        [0.5, 0.5],
        [0.75, 0.25],
        [1.0, 0.0]
    ])

    # 算法得到的前沿（略有偏差）
    obtained_front = np.array([
        [0.0, 0.95],
        [0.3, 0.7],
        [0.5, 0.55],
        [0.7, 0.3],
        [0.95, 0.05]
    ])

    print("\n真实Pareto前沿:")
    print(true_front)

    print("\n获得的Pareto前沿:")
    print(obtained_front)

    # 计算指标
    gd = generational_distance(obtained_front, true_front)
    print(f"\nGenerational Distance (GD): {gd:.6f}")

    sp = spacing(obtained_front)
    print(f"Spacing (SP): {sp:.6f}")

    igd = inverted_generational_distance(obtained_front, true_front)
    print(f"Inverted Generational Distance (IGD): {igd:.6f}")

    # 计算超体积
    reference_point = np.array([1.5, 1.5])
    hv = hypervolume(obtained_front, reference_point)
    print(f"Hypervolume (HV): {hv:.6f}")

    # 综合评估
    print("\n综合评估:")
    metrics = evaluate_performance(obtained_front, true_front, reference_point)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
