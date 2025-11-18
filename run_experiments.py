"""
运行MOAPO算法实验

对论文中的4个测试函数进行实验，并评估性能
"""

import numpy as np
import matplotlib.pyplot as plt
from moapo import MOAPO
from test_functions import get_test_function
from metrics import evaluate_performance, generational_distance, spacing
import time
import os


def run_single_experiment(test_func_name, n_runs=5, n_particles=50, n_iterations=100, verbose=False):
    """
    对单个测试函数运行多次实验

    Parameters:
    -----------
    test_func_name : str
        测试函数名称
    n_runs : int
        独立运行次数
    n_particles : int
        粒子数量
    n_iterations : int
        迭代次数
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    results : dict
        实验结果统计
    """
    print(f"\n{'='*60}")
    print(f"测试函数: {test_func_name.upper()}")
    print(f"{'='*60}")

    # 获取测试函数
    test_func = get_test_function(test_func_name)
    objectives = test_func.get_objectives()
    true_front = test_func.get_true_pareto_front(100)

    # 存储结果
    gd_values = []
    sp_values = []
    pareto_fronts = []
    execution_times = []

    # 运行多次实验
    for run in range(n_runs):
        if verbose:
            print(f"\n运行 {run + 1}/{n_runs}...")

        # 创建优化器
        moapo = MOAPO(
            n_objectives=test_func.n_objectives,
            n_particles=n_particles,
            n_iterations=n_iterations,
            bounds=test_func.bounds
        )

        # 运行优化
        start_time = time.time()
        pareto_front, pareto_fitness = moapo.optimize(objectives, verbose=verbose)
        end_time = time.time()

        execution_time = end_time - start_time
        execution_times.append(execution_time)

        # 转换为numpy数组
        pareto_fitness_array = np.array(pareto_fitness)

        # 计算性能指标
        gd = generational_distance(pareto_fitness_array, true_front)
        sp = spacing(pareto_fitness_array)

        gd_values.append(gd)
        sp_values.append(sp)
        pareto_fronts.append(pareto_fitness_array)

        if verbose:
            print(f"  找到 {len(pareto_front)} 个Pareto解")
            print(f"  GD: {gd:.6f}")
            print(f"  SP: {sp:.6f}")
            print(f"  执行时间: {execution_time:.2f}秒")

    # 统计结果
    results = {
        'test_function': test_func_name,
        'n_runs': n_runs,
        'gd': {
            'mean': np.mean(gd_values),
            'best': np.min(gd_values),
            'std': np.std(gd_values),
            'values': gd_values
        },
        'sp': {
            'mean': np.mean(sp_values),
            'best': np.min(sp_values),
            'std': np.std(sp_values),
            'values': sp_values
        },
        'execution_time': {
            'mean': np.mean(execution_times),
            'std': np.std(execution_times)
        },
        'pareto_fronts': pareto_fronts,
        'best_run_index': np.argmin(gd_values),  # 使用GD最小的运行作为最好结果
        'true_front': true_front
    }

    # 打印统计结果
    print(f"\n统计结果 (基于 {n_runs} 次运行):")
    print(f"  GD  - mean: {results['gd']['mean']:.6f}, "
          f"best: {results['gd']['best']:.6f}, "
          f"std: {results['gd']['std']:.6f}")
    print(f"  SP  - mean: {results['sp']['mean']:.6f}, "
          f"best: {results['sp']['best']:.6f}, "
          f"std: {results['sp']['std']:.6f}")
    print(f"  执行时间 - mean: {results['execution_time']['mean']:.2f}秒, "
          f"std: {results['execution_time']['std']:.2f}秒")

    return results


def plot_results(results, save_dir='results'):
    """
    绘制实验结果

    Parameters:
    -----------
    results : dict
        实验结果
    save_dir : str
        保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    test_func_name = results['test_function']
    best_idx = results['best_run_index']
    obtained_front = results['pareto_fronts'][best_idx]
    true_front = results['true_front']

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制真实Pareto前沿
    plt.plot(true_front[:, 0], true_front[:, 1], 'b-', linewidth=2,
             label='True Pareto Front', alpha=0.7)

    # 绘制MOAPO得到的Pareto前沿
    plt.scatter(obtained_front[:, 0], obtained_front[:, 1], c='red',
                marker='o', s=50, label='MOAPO', alpha=0.7, edgecolors='black')

    plt.xlabel('f1', fontsize=12)
    plt.ylabel('f2', fontsize=12)
    plt.title(f'{test_func_name.upper()} - Pareto Front Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图形
    save_path = os.path.join(save_dir, f'{test_func_name}_pareto_front.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图形已保存到: {save_path}")

    plt.close()


def run_all_experiments(n_runs=5, n_particles=50, n_iterations=100):
    """
    运行所有测试函数的实验

    Parameters:
    -----------
    n_runs : int
        每个测试函数的独立运行次数
    n_particles : int
        粒子数量
    n_iterations : int
        迭代次数
    """
    test_functions = ['schaffer1', 'zdt1', 'zdt2', 'zdt3']

    all_results = {}

    for test_func_name in test_functions:
        results = run_single_experiment(
            test_func_name,
            n_runs=n_runs,
            n_particles=n_particles,
            n_iterations=n_iterations,
            verbose=False
        )
        all_results[test_func_name] = results

        # 绘制结果
        plot_results(results)

    # 打印综合结果表
    print(f"\n{'='*80}")
    print("综合实验结果")
    print(f"{'='*80}")
    print(f"{'测试函数':<15} {'GD (mean)':<15} {'GD (best)':<15} {'SP (mean)':<15} {'SP (best)':<15}")
    print("-" * 80)

    for test_func_name in test_functions:
        results = all_results[test_func_name]
        print(f"{test_func_name.upper():<15} "
              f"{results['gd']['mean']:<15.6f} "
              f"{results['gd']['best']:<15.6f} "
              f"{results['sp']['mean']:<15.6f} "
              f"{results['sp']['best']:<15.6f}")

    print("=" * 80)

    return all_results


def main():
    """主函数"""
    print("MOAPO算法实验")
    print("=" * 60)
    print("参数设置:")
    print("  粒子数: 50")
    print("  迭代次数: 100")
    print("  独立运行次数: 5")
    print("=" * 60)

    # 运行所有实验
    results = run_all_experiments(
        n_runs=5,
        n_particles=50,
        n_iterations=100
    )

    print("\n实验完成！")
    print("结果已保存到 results/ 目录")


if __name__ == "__main__":
    main()
