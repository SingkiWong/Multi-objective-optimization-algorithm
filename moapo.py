"""
MOAPO: Multi-Objective Artificial Physics Optimization
基于拟态物理学优化的多目标优化算法

参考论文：
王艳, 曾建潮. 一种基于拟态物理学优化的多目标优化算法[J].
控制与决策, 2010, 25(7): 1040-1044.
"""

import numpy as np
from typing import List, Callable, Tuple
import copy


class MOAPO:
    """
    基于拟态物理学优化的多目标优化算法

    Parameters:
    -----------
    n_objectives : int
        目标函数数量
    n_particles : int
        粒子数量（群体规模）
    n_iterations : int
        最大迭代次数
    bounds : List[Tuple[float, float]]
        决策变量的边界 [(min1, max1), (min2, max2), ...]
    w_initial : float
        惯性权重初始值，默认0.9
    w_final : float
        惯性权重终值，默认0.4
    G_initial : float
        引力因子初始值，默认100
    G_final : float
        引力因子终值，默认1
    """

    def __init__(self,
                 n_objectives: int,
                 n_particles: int,
                 n_iterations: int,
                 bounds: List[Tuple[float, float]],
                 w_initial: float = 0.9,
                 w_final: float = 0.4,
                 G_initial: float = 100.0,
                 G_final: float = 1.0):

        self.n_objectives = n_objectives
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = np.array(bounds)
        self.n_dim = len(bounds)

        self.w_initial = w_initial
        self.w_final = w_final
        self.G_initial = G_initial
        self.G_final = G_final

        # 初始化粒子位置和速度
        self.positions = None
        self.velocities = None
        self.fitness = None  # shape: (n_particles, n_objectives)

        # 最好和最差解的索引
        self.best_indices = None  # 每个目标的最好解索引
        self.worst_indices = None  # 每个目标的最差解索引

        # Pareto前沿解集
        self.pareto_front = []
        self.pareto_fitness = []

    def initialize(self):
        """初始化粒子群"""
        # 随机初始化位置
        self.positions = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.n_particles, self.n_dim)
        )

        # 计算速度边界
        v_max = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.2

        # 随机初始化速度
        self.velocities = np.random.uniform(
            -v_max,
            v_max,
            (self.n_particles, self.n_dim)
        )

        self.v_max = v_max

    def evaluate(self, objective_funcs: List[Callable]):
        """
        评估所有粒子的适应值

        Parameters:
        -----------
        objective_funcs : List[Callable]
            目标函数列表
        """
        self.fitness = np.zeros((self.n_particles, self.n_objectives))

        for i in range(self.n_particles):
            for j, func in enumerate(objective_funcs):
                self.fitness[i, j] = func(self.positions[i])

    def compute_aggregated_fitness(self):
        """
        使用随机权重聚集方法计算聚合适应值

        Returns:
        --------
        aggregated_fitness : np.ndarray
            聚合后的适应值 (n_particles,)
        """
        # 为每个目标生成随机权重
        weights = np.random.random(self.n_objectives)
        weights = weights / np.sum(weights)  # 归一化

        # 计算加权和
        aggregated_fitness = np.dot(self.fitness, weights)

        return aggregated_fitness

    def find_global_best_worst(self):
        """
        找到每个目标的全局最好和最差解，并计算聚合的全局最好最差值

        Returns:
        --------
        f_best : float
            聚合的全局最好适应值
        f_worst : float
            聚合的全局最差适应值
        gbest : np.ndarray
            全局最好解位置
        gworst : np.ndarray
            全局最差解位置
        """
        # 找到每个目标的最好和最差索引（假设最小化）
        self.best_indices = np.argmin(self.fitness, axis=0)
        self.worst_indices = np.argmax(self.fitness, axis=0)

        # 获取每个目标的最好和最差适应值
        best_fitness = np.array([self.fitness[self.best_indices[i], i]
                                 for i in range(self.n_objectives)])
        worst_fitness = np.array([self.fitness[self.worst_indices[i], i]
                                  for i in range(self.n_objectives)])

        # 为每个目标的最好解生成随机权重并聚合
        weights_best = np.random.random(self.n_objectives)
        weights_best = weights_best / np.sum(weights_best)

        weights_worst = np.random.random(self.n_objectives)
        weights_worst = weights_worst / np.sum(weights_worst)

        # 聚合全局最好和最差位置
        gbest = np.zeros(self.n_dim)
        gworst = np.zeros(self.n_dim)

        for i in range(self.n_objectives):
            gbest += weights_best[i] * self.positions[self.best_indices[i]]
            gworst += weights_worst[i] * self.positions[self.worst_indices[i]]

        # 计算聚合后的全局最好和最差适应值
        f_best = np.dot(best_fitness, weights_best)
        f_worst = np.dot(worst_fitness, weights_worst)

        return f_best, f_worst, gbest, gworst

    def compute_mass(self, aggregated_fitness: np.ndarray,
                     f_best: float, f_worst: float):
        """
        计算粒子质量

        Parameters:
        -----------
        aggregated_fitness : np.ndarray
            聚合适应值
        f_best : float
            全局最好适应值
        f_worst : float
            全局最差适应值

        Returns:
        --------
        mass : np.ndarray
            粒子质量
        """
        # 避免除零
        if abs(f_worst - f_best) < 1e-10:
            return np.ones(self.n_particles)

        # 根据公式(1)计算质量
        mass = np.exp((f_best - aggregated_fitness) / (f_worst - f_best))

        return mass

    def compute_forces(self, mass: np.ndarray,
                       aggregated_fitness: np.ndarray,
                       best_idx: int, G: float):
        """
        计算粒子受到的虚拟力

        Parameters:
        -----------
        mass : np.ndarray
            粒子质量
        aggregated_fitness : np.ndarray
            聚合适应值
        best_idx : int
            当前最好解的索引
        G : float
            引力因子

        Returns:
        --------
        forces : np.ndarray
            每个粒子在各维度上受到的总力 (n_particles, n_dim)
        """
        forces = np.zeros((self.n_particles, self.n_dim))

        for i in range(self.n_particles):
            if i == best_idx:
                # 全局最优个体不受力
                continue

            for j in range(self.n_particles):
                if i == j:
                    continue

                # 计算距离向量
                distance_vector = self.positions[j] - self.positions[i]

                # 根据适应值判断吸引或排斥
                if aggregated_fitness[i] > aggregated_fitness[j]:
                    # i比j差，j吸引i（引力）
                    force = G * mass[i] * mass[j] * distance_vector
                else:
                    # i比j好，j排斥i（斥力）
                    force = -G * mass[i] * mass[j] * distance_vector

                forces[i] += force

        return forces

    def update_parameters(self, iteration: int):
        """
        动态更新惯性权重和引力因子

        Parameters:
        -----------
        iteration : int
            当前迭代次数

        Returns:
        --------
        w : float
            惯性权重
        G : float
            引力因子
        """
        # 计算w（前3/4迭代线性下降，之后保持）
        stop_iter = int(self.n_iterations * 3 / 4)

        if iteration < stop_iter:
            w = self.w_initial - (iteration / stop_iter) * (self.w_initial - self.w_final)
        else:
            w = self.w_final

        # 计算G（线性下降）
        G = self.G_initial - (iteration / self.n_iterations) * (self.G_initial - self.G_final)

        return w, G

    def update_velocity_position(self, forces: np.ndarray,
                                 mass: np.ndarray, w: float):
        """
        更新粒子速度和位置

        Parameters:
        -----------
        forces : np.ndarray
            粒子受到的力
        mass : np.ndarray
            粒子质量
        w : float
            惯性权重
        """
        # 生成随机数λ
        lambda_vals = np.random.uniform(0, 1, (self.n_particles, self.n_dim))

        # 更新速度（公式4）
        for i in range(self.n_particles):
            if mass[i] > 1e-10:  # 避免除零
                self.velocities[i] = w * self.velocities[i] + lambda_vals[i] * forces[i] / mass[i]

        # 限制速度
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)

        # 更新位置（公式5）
        self.positions = self.positions + self.velocities

        # 限制位置在边界内
        self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])

    def update_pareto_front(self):
        """更新Pareto前沿解集"""
        # 合并当前解和已有Pareto解
        all_positions = list(self.positions)
        all_fitness = list(self.fitness)

        if len(self.pareto_front) > 0:
            all_positions.extend(self.pareto_front)
            all_fitness.extend(self.pareto_fitness)

        all_positions = np.array(all_positions)
        all_fitness = np.array(all_fitness)

        # 找出非支配解
        pareto_indices = []
        n_solutions = len(all_fitness)

        for i in range(n_solutions):
            dominated = False
            for j in range(n_solutions):
                if i == j:
                    continue
                # 检查i是否被j支配
                if self._dominates(all_fitness[j], all_fitness[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)

        # 更新Pareto前沿
        self.pareto_front = [all_positions[i] for i in pareto_indices]
        self.pareto_fitness = [all_fitness[i] for i in pareto_indices]

    def _dominates(self, fitness1, fitness2):
        """
        判断fitness1是否支配fitness2（假设最小化）

        Parameters:
        -----------
        fitness1 : np.ndarray
            第一个解的适应值
        fitness2 : np.ndarray
            第二个解的适应值

        Returns:
        --------
        bool : 如果fitness1支配fitness2返回True
        """
        # fitness1至少在一个目标上更好，且在所有目标上不差于fitness2
        return np.all(fitness1 <= fitness2) and np.any(fitness1 < fitness2)

    def optimize(self, objective_funcs: List[Callable], verbose: bool = True):
        """
        执行MOAPO优化

        Parameters:
        -----------
        objective_funcs : List[Callable]
            目标函数列表
        verbose : bool
            是否打印优化过程

        Returns:
        --------
        pareto_front : List[np.ndarray]
            Pareto前沿解集（位置）
        pareto_fitness : List[np.ndarray]
            Pareto前沿解集（适应值）
        """
        # 初始化
        self.initialize()

        # 开始迭代
        for iteration in range(self.n_iterations):
            # 步骤1: 评估所有粒子
            self.evaluate(objective_funcs)

            # 步骤2: 计算聚合适应值
            aggregated_fitness = self.compute_aggregated_fitness()

            # 步骤3: 找到全局最好和最差
            f_best, f_worst, gbest, gworst = self.find_global_best_worst()

            # 步骤4: 计算质量
            mass = self.compute_mass(aggregated_fitness, f_best, f_worst)

            # 步骤5: 更新参数
            w, G = self.update_parameters(iteration)

            # 步骤6: 计算力
            best_idx = np.argmin(aggregated_fitness)
            forces = self.compute_forces(mass, aggregated_fitness, best_idx, G)

            # 步骤7: 更新速度和位置
            self.update_velocity_position(forces, mass, w)

            # 步骤8: 更新Pareto前沿
            self.update_pareto_front()

            if verbose and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                print(f"Iteration {iteration + 1}/{self.n_iterations}, "
                      f"Pareto solutions: {len(self.pareto_front)}, "
                      f"w: {w:.3f}, G: {G:.3f}")

        return self.pareto_front, self.pareto_fitness


if __name__ == "__main__":
    # 简单测试
    def schaffer1_f1(x):
        return x[0] ** 2

    def schaffer1_f2(x):
        return (x[0] - 2) ** 2

    # 定义问题
    bounds = [(-10, 10)]
    moapo = MOAPO(
        n_objectives=2,
        n_particles=50,
        n_iterations=100,
        bounds=bounds
    )

    # 优化
    pareto_front, pareto_fitness = moapo.optimize([schaffer1_f1, schaffer1_f2])

    print(f"\n找到 {len(pareto_front)} 个Pareto最优解")
    print(f"目标值范围: f1=[{min(f[0] for f in pareto_fitness):.3f}, {max(f[0] for f in pareto_fitness):.3f}], "
          f"f2=[{min(f[1] for f in pareto_fitness):.3f}, {max(f[1] for f in pareto_fitness):.3f}]")
