"""
多目标优化测试函数

包含论文中使用的测试函数：
- Schaffer1
- ZDT1
- ZDT2
- ZDT3
"""

import numpy as np
from typing import Tuple, List


class TestFunction:
    """测试函数基类"""

    def __init__(self, n_objectives: int, bounds: List[Tuple[float, float]]):
        self.n_objectives = n_objectives
        self.bounds = bounds
        self.n_dim = len(bounds)

    def get_objectives(self):
        """返回目标函数列表"""
        raise NotImplementedError

    def get_true_pareto_front(self, n_points: int = 100):
        """获取真实的Pareto前沿（用于评估）"""
        raise NotImplementedError


class Schaffer1(TestFunction):
    """
    Schaffer1 测试函数

    目标函数:
    f1(x) = x^2
    f2(x) = (x-2)^2

    决策变量: x ∈ [-10, 10]
    Pareto最优解: x ∈ [0, 2]
    """

    def __init__(self):
        super().__init__(n_objectives=2, bounds=[(-10, 10)])

    def f1(self, x):
        """第一个目标函数"""
        return x[0] ** 2

    def f2(self, x):
        """第二个目标函数"""
        return (x[0] - 2) ** 2

    def get_objectives(self):
        """返回目标函数列表"""
        return [self.f1, self.f2]

    def get_true_pareto_front(self, n_points: int = 100):
        """
        获取真实的Pareto前沿

        Returns:
        --------
        pareto_front : np.ndarray
            Pareto前沿的目标值 (n_points, 2)
        """
        x = np.linspace(0, 2, n_points)
        f1_values = x ** 2
        f2_values = (x - 2) ** 2
        return np.column_stack([f1_values, f2_values])


class ZDT1(TestFunction):
    """
    ZDT1 测试函数

    目标函数:
    f1(x) = x1
    g(x) = 1 + 9 * sum(xi) / (n-1), i=2 to n
    h(f1, g) = 1 - sqrt(f1/g)
    f2(x) = g(x) * h(f1, g)

    决策变量: xi ∈ [0, 1], i=1 to n
    Pareto最优解: x1 ∈ [0, 1], xi = 0, i=2 to n
    """

    def __init__(self, n_dim: int = 30):
        bounds = [(0, 1) for _ in range(n_dim)]
        super().__init__(n_objectives=2, bounds=bounds)

    def f1(self, x):
        """第一个目标函数"""
        return x[0]

    def g(self, x):
        """辅助函数g"""
        n = len(x)
        if n == 1:
            return 1.0
        return 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)

    def h(self, f1, g):
        """辅助函数h"""
        return 1.0 - np.sqrt(f1 / g)

    def f2(self, x):
        """第二个目标函数"""
        f1_val = self.f1(x)
        g_val = self.g(x)
        h_val = self.h(f1_val, g_val)
        return g_val * h_val

    def get_objectives(self):
        """返回目标函数列表"""
        return [self.f1, self.f2]

    def get_true_pareto_front(self, n_points: int = 100):
        """
        获取真实的Pareto前沿

        Returns:
        --------
        pareto_front : np.ndarray
            Pareto前沿的目标值 (n_points, 2)
        """
        f1_values = np.linspace(0, 1, n_points)
        f2_values = 1.0 - np.sqrt(f1_values)
        return np.column_stack([f1_values, f2_values])


class ZDT2(TestFunction):
    """
    ZDT2 测试函数

    目标函数:
    f1(x) = x1
    g(x) = 1 + 9 * sum(xi) / (n-1), i=2 to n
    h(f1, g) = 1 - (f1/g)^2
    f2(x) = g(x) * h(f1, g)

    决策变量: xi ∈ [0, 1], i=1 to n
    Pareto最优解: x1 ∈ [0, 1], xi = 0, i=2 to n
    Pareto前沿: 非凸
    """

    def __init__(self, n_dim: int = 30):
        bounds = [(0, 1) for _ in range(n_dim)]
        super().__init__(n_objectives=2, bounds=bounds)

    def f1(self, x):
        """第一个目标函数"""
        return x[0]

    def g(self, x):
        """辅助函数g"""
        n = len(x)
        if n == 1:
            return 1.0
        return 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)

    def h(self, f1, g):
        """辅助函数h"""
        return 1.0 - (f1 / g) ** 2

    def f2(self, x):
        """第二个目标函数"""
        f1_val = self.f1(x)
        g_val = self.g(x)
        h_val = self.h(f1_val, g_val)
        return g_val * h_val

    def get_objectives(self):
        """返回目标函数列表"""
        return [self.f1, self.f2]

    def get_true_pareto_front(self, n_points: int = 100):
        """
        获取真实的Pareto前沿

        Returns:
        --------
        pareto_front : np.ndarray
            Pareto前沿的目标值 (n_points, 2)
        """
        f1_values = np.linspace(0, 1, n_points)
        f2_values = 1.0 - f1_values ** 2
        return np.column_stack([f1_values, f2_values])


class ZDT3(TestFunction):
    """
    ZDT3 测试函数

    目标函数:
    f1(x) = x1
    g(x) = 1 + 9 * sum(xi) / (n-1), i=2 to n
    h(f1, g) = 1 - sqrt(f1/g) - (f1/g) * sin(10*pi*f1)
    f2(x) = g(x) * h(f1, g)

    决策变量: xi ∈ [0, 1], i=1 to n
    Pareto最优解: x1 ∈ [0, 1], xi = 0, i=2 to n
    Pareto前沿: 不连续
    """

    def __init__(self, n_dim: int = 30):
        bounds = [(0, 1) for _ in range(n_dim)]
        super().__init__(n_objectives=2, bounds=bounds)

    def f1(self, x):
        """第一个目标函数"""
        return x[0]

    def g(self, x):
        """辅助函数g"""
        n = len(x)
        if n == 1:
            return 1.0
        return 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)

    def h(self, f1, g):
        """辅助函数h"""
        return 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)

    def f2(self, x):
        """第二个目标函数"""
        f1_val = self.f1(x)
        g_val = self.g(x)
        h_val = self.h(f1_val, g_val)
        return g_val * h_val

    def get_objectives(self):
        """返回目标函数列表"""
        return [self.f1, self.f2]

    def get_true_pareto_front(self, n_points: int = 100):
        """
        获取真实的Pareto前沿（不连续）

        Returns:
        --------
        pareto_front : np.ndarray
            Pareto前沿的目标值 (n_points, 2)
        """
        # ZDT3的真实Pareto前沿是不连续的
        # 这里生成密集采样点，包含不连续区域
        f1_values = np.linspace(0, 1, n_points)
        f2_values = 1.0 - np.sqrt(f1_values) - f1_values * np.sin(10 * np.pi * f1_values)

        # 只保留非支配的点
        pareto_front = []
        for i in range(len(f1_values)):
            f1, f2 = f1_values[i], f2_values[i]
            # 检查是否被支配
            dominated = False
            for j in range(len(f1_values)):
                if i == j:
                    continue
                if f1_values[j] <= f1 and f2_values[j] <= f2 and (f1_values[j] < f1 or f2_values[j] < f2):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append([f1, f2])

        return np.array(pareto_front) if len(pareto_front) > 0 else np.column_stack([f1_values, f2_values])


def get_test_function(name: str, n_dim: int = 30):
    """
    根据名称获取测试函数

    Parameters:
    -----------
    name : str
        测试函数名称: 'schaffer1', 'zdt1', 'zdt2', 'zdt3'
    n_dim : int
        决策变量维度（仅用于ZDT函数）

    Returns:
    --------
    test_func : TestFunction
        测试函数对象
    """
    name = name.lower()

    if name == 'schaffer1':
        return Schaffer1()
    elif name == 'zdt1':
        return ZDT1(n_dim)
    elif name == 'zdt2':
        return ZDT2(n_dim)
    elif name == 'zdt3':
        return ZDT3(n_dim)
    else:
        raise ValueError(f"未知的测试函数: {name}")


if __name__ == "__main__":
    # 测试所有函数
    print("测试函数定义:")
    print("-" * 50)

    for func_name in ['schaffer1', 'zdt1', 'zdt2', 'zdt3']:
        test_func = get_test_function(func_name)
        print(f"\n{func_name.upper()}:")
        print(f"  目标数: {test_func.n_objectives}")
        print(f"  维度: {test_func.n_dim}")
        print(f"  边界: {test_func.bounds[:3]}..." if len(test_func.bounds) > 3 else f"  边界: {test_func.bounds}")

        # 测试真实Pareto前沿
        true_pf = test_func.get_true_pareto_front(10)
        print(f"  真实Pareto前沿点数: {len(true_pf)}")
        print(f"  前3个点: {true_pf[:3].tolist()}")
