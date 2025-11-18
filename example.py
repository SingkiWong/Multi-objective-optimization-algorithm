"""
MOAPO算法使用示例

这个文件展示了如何使用MOAPO算法解决多目标优化问题
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from moapo import MOAPO
from imoapo import IMOAPO
from test_functions import get_test_function
from metrics import evaluate_performance


def example_1_simple_usage():
    """示例1：简单使用"""
    print("=" * 60)
    print("示例1：简单使用MOAPO算法")
    print("=" * 60)

    # 定义目标函数
    def f1(x):
        return x[0] ** 2

    def f2(x):
        return (x[0] - 2) ** 2

    # 创建优化器
    moapo = MOAPO(
        n_objectives=2,
        n_particles=50,
        n_iterations=100,
        bounds=[(-10, 10)]
    )

    # 运行优化
    pareto_front, pareto_fitness = moapo.optimize([f1, f2], verbose=False)

    print(f"\n找到 {len(pareto_front)} 个Pareto最优解")
    print(f"决策变量范围: [{min(p[0] for p in pareto_front):.3f}, "
          f"{max(p[0] for p in pareto_front):.3f}]")
    print(f"目标值范围: f1=[{min(f[0] for f in pareto_fitness):.3f}, "
          f"{max(f[0] for f in pareto_fitness):.3f}], "
          f"f2=[{min(f[1] for f in pareto_fitness):.3f}, "
          f"{max(f[1] for f in pareto_fitness):.3f}]")


def example_2_with_test_function():
    """示例2：使用预定义的测试函数"""
    print("\n" + "=" * 60)
    print("示例2：使用预定义的测试函数")
    print("=" * 60)

    # 获取ZDT1测试函数
    test_func = get_test_function('zdt1', n_dim=10)  # 使用10维简化计算
    objectives = test_func.get_objectives()

    # 创建优化器
    moapo = MOAPO(
        n_objectives=test_func.n_objectives,
        n_particles=50,
        n_iterations=100,
        bounds=test_func.bounds
    )

    # 运行优化
    print("\n正在优化ZDT1函数...")
    pareto_front, pareto_fitness = moapo.optimize(objectives, verbose=False)

    # 获取真实Pareto前沿
    true_front = test_func.get_true_pareto_front(100)

    # 评估性能
    pareto_fitness_array = np.array(pareto_fitness)
    metrics = evaluate_performance(pareto_fitness_array, true_front)

    print(f"\n找到 {len(pareto_front)} 个Pareto最优解")
    print(f"性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def example_3_parameter_tuning():
    """示例3：参数调优"""
    print("\n" + "=" * 60)
    print("示例3：参数调优对比")
    print("=" * 60)

    test_func = get_test_function('schaffer1')
    objectives = test_func.get_objectives()
    true_front = test_func.get_true_pareto_front(100)

    # 测试不同参数设置
    configs = [
        {'name': '默认参数', 'n_particles': 50, 'G_initial': 100},
        {'name': '大群体', 'n_particles': 100, 'G_initial': 100},
        {'name': '强引力', 'n_particles': 50, 'G_initial': 200},
    ]

    results = []

    for config in configs:
        print(f"\n测试配置: {config['name']}")
        print(f"  粒子数: {config['n_particles']}, 初始引力因子: {config['G_initial']}")

        moapo = MOAPO(
            n_objectives=2,
            n_particles=config['n_particles'],
            n_iterations=50,
            bounds=test_func.bounds,
            G_initial=config['G_initial']
        )

        pareto_front, pareto_fitness = moapo.optimize(objectives, verbose=False)
        pareto_fitness_array = np.array(pareto_fitness)

        metrics = evaluate_performance(pareto_fitness_array, true_front)

        print(f"  找到解的数量: {len(pareto_front)}")
        print(f"  GD: {metrics['GD']:.6f}")
        print(f"  SP: {metrics['SP']:.6f}")

        results.append({
            'config': config['name'],
            'n_solutions': len(pareto_front),
            'GD': metrics['GD'],
            'SP': metrics['SP']
        })

    # 打印对比表
    print("\n参数对比总结:")
    print(f"{'配置':<15} {'解数量':<10} {'GD':<12} {'SP':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['config']:<15} {r['n_solutions']:<10} {r['GD']:<12.6f} {r['SP']:<12.6f}")


def example_4_visualization():
    """示例4：可视化结果"""
    print("\n" + "=" * 60)
    print("示例4：可视化Pareto前沿")
    print("=" * 60)

    # 对多个测试函数生成可视化
    test_functions = ['schaffer1', 'zdt1', 'zdt2', 'zdt3']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, func_name in enumerate(test_functions):
        print(f"\n处理 {func_name.upper()}...")

        test_func = get_test_function(func_name, n_dim=10 if 'zdt' in func_name else 1)
        objectives = test_func.get_objectives()
        true_front = test_func.get_true_pareto_front(100)

        # 运行MOAPO
        moapo = MOAPO(
            n_objectives=2,
            n_particles=50,
            n_iterations=50,
            bounds=test_func.bounds
        )

        pareto_front, pareto_fitness = moapo.optimize(objectives, verbose=False)
        pareto_fitness_array = np.array(pareto_fitness)

        # 绘图
        ax = axes[idx]
        ax.plot(true_front[:, 0], true_front[:, 1], 'b-', linewidth=2,
                label='True PF', alpha=0.6)
        ax.scatter(pareto_fitness_array[:, 0], pareto_fitness_array[:, 1],
                   c='red', marker='o', s=30, label='MOAPO', alpha=0.7)
        ax.set_xlabel('f1', fontsize=10)
        ax.set_ylabel('f2', fontsize=10)
        ax.set_title(f'{func_name.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('moapo_comparison.png', dpi=200, bbox_inches='tight')
    print("\n可视化结果已保存到: moapo_comparison.png")


def example_5_vrp_imoapo():
    """示例5：使用IMOAPO解决小规模车辆路径规划问题"""

    print("\n" + "=" * 60)
    print("示例5：IMOAPO 求解车辆路径规划")
    print("=" * 60)

    # 构造一个5个客户的简化VRP (节点0为仓库)
    distance_matrix = np.array(
        [
            [0, 6, 9, 7, 3, 5],
            [6, 0, 5, 3, 7, 4],
            [9, 5, 0, 4, 8, 6],
            [7, 3, 4, 0, 6, 5],
            [3, 7, 8, 6, 0, 4],
            [5, 4, 6, 5, 4, 0],
        ],
        dtype=float,
    )

    # 客户需求 (共5个客户，对应索引1-5)
    demands = [1.5, 1.0, 1.7, 0.9, 1.2]

    optimizer = IMOAPO(
        distance_matrix=distance_matrix,
        demands=demands,
        vehicle_capacity=3.0,
        n_vehicles=3,
        n_particles=40,
        n_iterations=80,
    )

    pareto_front, pareto_fitness = optimizer.optimize(verbose=False)

    # 选择加权和最优解用于演示
    scores = np.sum(pareto_fitness, axis=1)
    best_idx = int(np.argmin(scores))

    print(f"找到 {len(pareto_front)} 个Pareto候选解，展示其中一个：\n")
    print(optimizer.summarize_solution(pareto_front[best_idx]))


def main():
    """运行所有示例"""
    print("\nMOAPO算法使用示例\n")

    # 运行示例
    example_1_simple_usage()
    example_2_with_test_function()
    example_3_parameter_tuning()
    example_4_visualization()
    example_5_vrp_imoapo()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
