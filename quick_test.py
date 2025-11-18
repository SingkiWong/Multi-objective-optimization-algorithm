"""快速测试MOAPO算法"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from moapo import MOAPO
from test_functions import get_test_function
from metrics import generational_distance, spacing

# 测试Schaffer1函数
print("快速测试：Schaffer1函数")
print("=" * 50)

test_func = get_test_function('schaffer1')
objectives = test_func.get_objectives()
true_front = test_func.get_true_pareto_front(100)

# 创建优化器
moapo = MOAPO(
    n_objectives=2,
    n_particles=30,
    n_iterations=50,
    bounds=test_func.bounds
)

# 运行优化
pareto_front, pareto_fitness = moapo.optimize(objectives, verbose=True)

# 转换为numpy数组
pareto_fitness_array = np.array(pareto_fitness)

# 计算性能指标
gd = generational_distance(pareto_fitness_array, true_front)
sp = spacing(pareto_fitness_array)

print(f"\n性能指标:")
print(f"  找到 {len(pareto_front)} 个Pareto解")
print(f"  GD: {gd:.6f}")
print(f"  SP: {sp:.6f}")

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(true_front[:, 0], true_front[:, 1], 'b-', linewidth=2,
         label='True Pareto Front', alpha=0.7)
plt.scatter(pareto_fitness_array[:, 0], pareto_fitness_array[:, 1],
            c='red', marker='o', s=50, label='MOAPO', alpha=0.7, edgecolors='black')
plt.xlabel('f1', fontsize=12)
plt.ylabel('f2', fontsize=12)
plt.title('Schaffer1 - Pareto Front', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('schaffer1_test.png', dpi=150)
print("\n图形已保存到: schaffer1_test.png")
