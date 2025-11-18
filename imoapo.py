"""
IMOAPO: Improved MOAPO tailored for vehicle routing problems.

本模块基于第三章提出的IMOAPO思想，针对车辆路径规划（如医疗废弃物
运输）构建多目标求解器。算法沿用MOAPO的物理启发式框架，同时结合
随机键编码的路线解码方式处理车辆容量约束和路径均衡目标。
"""

from typing import List, Tuple
import numpy as np

from moapo import MOAPO


class IMOAPO(MOAPO):
    """面向车辆路径规划的IMOAPO算法实现。

    相比通用MOAPO，本实现采用随机键（random keys）表示客户访问顺序，
    通过容量约束的贪心分割生成多辆车的运输路径。默认优化两个目标：

    1. **运输代价**：路线总距离，并对超载和车辆数超限施加强罚。
    2. **均衡性**：路线距离的离散程度，兼顾行驶效率与载重惩罚。

    Parameters
    ----------
    distance_matrix : np.ndarray
        包含配送中心（索引0）和所有客户的对称距离矩阵，形状为
        ``(n_customers + 1, n_customers + 1)``。
    demands : List[float]
        每个客户的需求量，长度为 ``n_customers``（不含仓库）。
    vehicle_capacity : float
        每辆车的容量上限。
    n_vehicles : int
        可用车辆数量上限。
    n_particles : int
        种群规模。
    n_iterations : int
        最大迭代次数。
    w_initial, w_final, G_initial, G_final : float
        IMOAPO沿用的惯性权重与引力因子设置。
    archive_size : int, optional
        外部档案容量，默认等于 ``n_particles``。
    capacity_penalty : float, optional
        对超载部分的惩罚系数，默认1000。
    vehicle_penalty : float, optional
        对车辆数量超限的惩罚系数，默认500。
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        demands: List[float],
        vehicle_capacity: float,
        n_vehicles: int,
        n_particles: int,
        n_iterations: int,
        w_initial: float = 0.9,
        w_final: float = 0.4,
        G_initial: float = 100.0,
        G_final: float = 1.0,
        archive_size: int | None = None,
        capacity_penalty: float = 1000.0,
        vehicle_penalty: float = 500.0,
    ) -> None:
        self.distance_matrix = np.asarray(distance_matrix, dtype=float)
        self.demands = np.asarray(demands, dtype=float)
        self.vehicle_capacity = float(vehicle_capacity)
        self.n_vehicles = int(n_vehicles)
        self.capacity_penalty = float(capacity_penalty)
        self.vehicle_penalty = float(vehicle_penalty)

        n_customers = len(self.demands)
        if self.distance_matrix.shape[0] != n_customers + 1:
            raise ValueError(
                "distance_matrix 大小与客户数量不匹配，应包含仓库+客户的节点"
            )

        # 随机键编码，决策变量维度与客户数量一致，范围[0, 1]
        bounds: List[Tuple[float, float]] = [(0.0, 1.0)] * n_customers

        super().__init__(
            n_objectives=2,
            n_particles=n_particles,
            n_iterations=n_iterations,
            bounds=bounds,
            w_initial=w_initial,
            w_final=w_final,
            G_initial=G_initial,
            G_final=G_final,
            archive_size=archive_size,
        )

    # --------------------------- 路线解码与评价 --------------------------- #
    def decode_routes(self, position: np.ndarray) -> List[List[int]]:
        """根据随机键位置向量生成车辆路径。

        返回的每条路线均以0为起点和终点，内部节点为客户编号(1-based)。
        """

        # 依据随机键排序得到访问序列（客户编号从1开始）
        visit_order = np.argsort(position) + 1

        routes: List[List[int]] = []
        current_route: List[int] = [0]
        current_load = 0.0

        for customer_idx in visit_order:
            demand = self.demands[customer_idx - 1]
            if current_load + demand > self.vehicle_capacity and len(routes) + 1 < self.n_vehicles:
                # 结束当前路线，开启新车
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
                current_load = 0.0

            current_route.append(customer_idx)
            current_load += demand

        # 关闭最后一条路线
        current_route.append(0)
        routes.append(current_route)

        return routes

    def _route_distance(self, route: List[int]) -> float:
        distance = 0.0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i], route[i + 1]]
        return distance

    def _route_load(self, route: List[int]) -> float:
        load = 0.0
        for node in route:
            if node == 0:
                continue
            load += self.demands[node - 1]
        return load

    def _evaluate_metrics(self, position: np.ndarray) -> dict:
        routes = self.decode_routes(position)
        route_distances = [self._route_distance(r) for r in routes]
        route_loads = [self._route_load(r) for r in routes]

        total_distance = float(np.sum(route_distances))
        overload = float(np.sum([max(0.0, load - self.vehicle_capacity) for load in route_loads]))
        vehicles_used = len(routes)
        distance_balance = float(np.std(route_distances)) if len(route_distances) > 1 else 0.0

        return {
            "routes": routes,
            "route_distances": route_distances,
            "total_distance": total_distance,
            "overload": overload,
            "vehicles_used": vehicles_used,
            "distance_balance": distance_balance,
        }

    # --------------------------- 算法主流程覆盖 --------------------------- #
    def evaluate(self) -> None:  # type: ignore[override]
        """评估所有粒子在VRP目标下的适应值。"""

        self.fitness = np.zeros((self.n_particles, self.n_objectives))

        for i in range(self.n_particles):
            metrics = self._evaluate_metrics(self.positions[i])

            # 目标1：总代价 + 容量/车辆惩罚
            penalty_overload = self.capacity_penalty * metrics["overload"]
            penalty_vehicle = self.vehicle_penalty * max(0, metrics["vehicles_used"] - self.n_vehicles)
            cost_obj = metrics["total_distance"] + penalty_overload + penalty_vehicle

            # 目标2：均衡性 + 轻量效率项
            balance_obj = (
                metrics["distance_balance"]
                + 0.01 * metrics["total_distance"]
                + 0.1 * metrics["overload"]
            )

            self.fitness[i, 0] = cost_obj
            self.fitness[i, 1] = balance_obj

    def optimize(self, verbose: bool = True):  # type: ignore[override]
        """执行IMOAPO优化流程。"""

        self.initialize()

        for iteration in range(self.n_iterations):
            # 评估适应值
            self.evaluate()

            # 聚合适应值 & 最优/最差
            aggregated_fitness, _ = self.compute_aggregated_fitness()
            f_best, f_worst, gbest, gworst, best_idx, worst_idx = self.find_global_best_worst(
                aggregated_fitness
            )

            # 质量、参数、力计算
            mass = self.compute_mass(aggregated_fitness, f_best, f_worst)
            w, G = self.update_parameters(iteration)
            forces = self.compute_forces(mass, aggregated_fitness, best_idx, G)

            # 更新粒子状态
            self.update_velocity_position(forces, mass, w)

            # 更新Pareto档案
            self.update_pareto_front()

            if verbose and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                print(
                    f"Iteration {iteration + 1}/{self.n_iterations}, "
                    f"Pareto solutions: {len(self.pareto_front)}, "
                    f"w: {w:.3f}, G: {G:.3f}"
                )

        return self.pareto_front, self.pareto_fitness

    # --------------------------- 辅助可视化 --------------------------- #
    def summarize_solution(self, position: np.ndarray) -> str:
        """返回给定位置向量对应路线的文本摘要，便于打印和调试。"""

        metrics = self._evaluate_metrics(position)
        lines = [
            f"车辆数: {metrics['vehicles_used']}, 总距离: {metrics['total_distance']:.2f}",
            f"超载量: {metrics['overload']:.2f}, 路径离散度: {metrics['distance_balance']:.2f}",
        ]

        for idx, (route, distance) in enumerate(zip(metrics["routes"], metrics["route_distances"])):
            lines.append(
                f"  车辆{idx + 1}: 路径 {route} | 距离 {distance:.2f} | 载重 {self._route_load(route):.2f}"
            )

        return "\n".join(lines)


__all__ = ["IMOAPO"]
