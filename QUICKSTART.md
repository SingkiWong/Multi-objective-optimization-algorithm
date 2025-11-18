# MOAPO 快速入门指南

## 一分钟快速开始

### 1. 安装依赖
```bash
pip install numpy matplotlib
```

### 2. 最简单的使用示例
```python
from moapo import MOAPO

# 定义两个冲突的目标函数
def f1(x):
    return x[0] ** 2

def f2(x):
    return (x[0] - 2) ** 2

# 创建优化器并运行
moapo = MOAPO(
    n_objectives=2,          # 2个目标
    n_particles=50,          # 50个粒子
    n_iterations=100,        # 100次迭代
    bounds=[(-10, 10)]       # 决策变量范围
)

# 优化并获取Pareto前沿
pareto_front, pareto_fitness = moapo.optimize([f1, f2])

print(f"找到 {len(pareto_front)} 个Pareto最优解")
```

### 3. 运行测试
```bash
# 快速测试
python quick_test.py

# 完整示例
python example.py

# 完整实验（需要较长时间）
python run_experiments.py
```

## 核心概念

### 什么是MOAPO？
MOAPO是一种基于物理学原理的多目标优化算法：
- **粒子系统**：每个粒子代表一个候选解
- **虚拟力**：好的粒子吸引差的粒子，差的粒子排斥好的粒子
- **多目标处理**：使用随机权重聚集多个目标
- **动态参数**：w和G随迭代自动调整

### 关键参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| n_particles | 30-100 | 简单问题用30-50，复杂问题用50-100 |
| n_iterations | 50-200 | 根据问题复杂度调整 |
| w_initial | 0.9 | 惯性权重初始值（建议保持默认） |
| G_initial | 100 | 引力因子初始值（建议保持默认） |

## 实用技巧

### 1. 如何判断算法效果？
使用性能指标：
```python
from metrics import evaluate_performance

metrics = evaluate_performance(obtained_front, true_front)
print(f"GD: {metrics['GD']:.6f}")  # 越小越好
print(f"SP: {metrics['SP']:.6f}")  # 越小越好
```

### 2. 如何可视化结果？
```python
import matplotlib.pyplot as plt
import numpy as np

pareto_fitness_array = np.array(pareto_fitness)
plt.scatter(pareto_fitness_array[:, 0], pareto_fitness_array[:, 1])
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Front')
plt.show()
```

### 3. 常见问题

**Q: 收敛太慢？**
- 增加粒子数量
- 增大G_initial（如200）

**Q: 收敛太快，陷入局部最优？**
- 增大w_initial（如0.95）
- 减小G_initial（如50）

**Q: Pareto前沿分布不均匀？**
- 增加迭代次数
- 增加粒子数量
- 多次运行取最好结果

## 测试函数

### Schaffer1（入门级）
```python
from test_functions import get_test_function

test_func = get_test_function('schaffer1')
objectives = test_func.get_objectives()
# 1维决策变量，简单快速
```

### ZDT系列（标准测试）
```python
# ZDT1: 凸Pareto前沿
test_func = get_test_function('zdt1', n_dim=30)

# ZDT2: 非凸Pareto前沿
test_func = get_test_function('zdt2', n_dim=30)

# ZDT3: 不连续Pareto前沿
test_func = get_test_function('zdt3', n_dim=30)
```

## 预期性能

在标准测试上的典型结果：

| 函数 | GD | SP | 运行时间 |
|------|-----|-----|---------|
| Schaffer1 | ~0.002 | ~0.03 | < 5秒 |
| ZDT1 | ~0.02 | ~0.01 | < 30秒 |
| ZDT2 | ~0.05 | ~0.01 | < 30秒 |
| ZDT3 | ~0.02 | ~0.01 | < 30秒 |

## 进阶使用

### 自定义多目标问题
```python
def objective1(x):
    # 你的目标函数1
    return some_calculation(x)

def objective2(x):
    # 你的目标函数2
    return another_calculation(x)

def objective3(x):
    # 你的目标函数3
    return yet_another_calculation(x)

moapo = MOAPO(
    n_objectives=3,
    n_particles=100,
    n_iterations=200,
    bounds=[(-10, 10)] * 5  # 5维决策空间
)

pareto_front, pareto_fitness = moapo.optimize(
    [objective1, objective2, objective3]
)
```

## 参考资料

- 论文：王艳, 曾建潮. 一种基于拟态物理学优化的多目标优化算法[J]. 控制与决策, 2010, 25(7): 1040-1044.
- README.md：完整文档
- example.py：更多示例代码

## 获取帮助

如有问题，请查看：
1. README.md 中的详细说明
2. example.py 中的多个使用示例
3. 源代码中的详细注释
