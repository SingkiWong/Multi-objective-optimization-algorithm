# MOAPO：基于拟态物理学优化的多目标优化算法

这是对论文《一种基于拟态物理学优化的多目标优化算法》（王艳，曾建潮，2010）的Python实现。

## 算法简介

MOAPO (Multi-Objective Artificial Physics Optimization) 是一种基于拟态物理学原理的多目标优化算法。该算法受牛顿第二定律启发，通过个体间的虚拟力作用来调整粒子的运动，从而搜索多目标优化问题的Pareto最优解集。

### 核心特点

- **物理启发**：模拟粒子间的引力和斥力
- **多目标处理**：使用随机权重聚集方法处理多个目标
- **动态参数**：惯性权重和引力因子随迭代动态调整
- **良好的分布性**：能够获得分布均匀的Pareto前沿
- **外部档案**：维护非支配解档案，并用拥挤距离截断保证多样性

## 算法原理

### 1. 个体质量计算

```
m_i = exp((f(x_best) - f(x_i)) / (f(x_worst) - f(x_best)))
```

### 2. 虚拟力计算

- **引力**（当f(x_i) > f(x_j)时）：
  ```
  F_ij,k = G * m_i * m_j * (x_j,k - x_i,k)
  ```

- **斥力**（当f(x_i) ≤ f(x_j)时）：
  ```
  F_ij,k = -G * m_i * m_j * (x_j,k - x_i,k)
  ```

### 3. 速度和位置更新

```
v_i,k(t+1) = w * v_i,k(t) + λ * F_i,k / m_i
x_i,k(t+1) = x_i,k(t) + v_i,k(t)
```

### 4. 动态参数调整

- **惯性权重w**：从0.9线性下降到0.4（前3/4迭代）
- **引力因子G**：从100线性下降到1
- **外部档案截断**：使用拥挤距离排序，超出容量时优先保留稀疏区域的解

## 项目结构

```
Multi-objective-optimization-algorithm/
├── moapo.py                 # MOAPO算法核心实现
├── test_functions.py        # 测试函数（Schaffer1, ZDT1, ZDT2, ZDT3）
├── metrics.py               # 性能评价指标（GD, SP, IGD, HV）
├── run_experiments.py       # 完整实验脚本
├── quick_test.py           # 快速测试脚本
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install numpy matplotlib
```

## 快速开始

### 1. 简单示例

```python
from moapo import MOAPO
from test_functions import get_test_function

# 获取测试函数
test_func = get_test_function('schaffer1')
objectives = test_func.get_objectives()

# 创建优化器
moapo = MOAPO(
    n_objectives=2,
    n_particles=50,
    n_iterations=100,
    bounds=test_func.bounds
)

# 运行优化
pareto_front, pareto_fitness = moapo.optimize(objectives)

print(f"找到 {len(pareto_front)} 个Pareto最优解")
```

### 2. 运行快速测试

```bash
python quick_test.py
```

### 3. 运行完整实验

```bash
python run_experiments.py
```

这将对所有测试函数（Schaffer1, ZDT1, ZDT2, ZDT3）运行实验，并生成性能统计和可视化结果。

## 测试函数

### Schaffer1
- **目标函数**：
  - f1(x) = x²
  - f2(x) = (x-2)²
- **决策变量**：x ∈ [-10, 10]
- **Pareto最优解**：x ∈ [0, 2]

### ZDT1
- **特点**：凸Pareto前沿
- **维度**：30维
- **决策变量**：x_i ∈ [0, 1]

### ZDT2
- **特点**：非凸Pareto前沿
- **维度**：30维
- **决策变量**：x_i ∈ [0, 1]

### ZDT3
- **特点**：不连续Pareto前沿
- **维度**：30维
- **决策变量**：x_i ∈ [0, 1]

## 性能评价指标

### GD (Generational Distance)
衡量算法得到的Pareto前沿与真实前沿的逼近程度。值越小越好。

```
GD = (1/n) * sqrt(Σ d_i²)
```

### SP (Spacing)
衡量算法得到的Pareto前沿的分布均匀性。值越小越好。

```
SP = sqrt((1/(n-1)) * Σ (d̄ - d_i)²)
```

### IGD (Inverted Generational Distance)
从真实前沿到算法得到前沿的距离，同时考虑收敛性和分布性。

### HV (Hypervolume)
衡量Pareto前沿覆盖的目标空间体积（仅支持2维目标）。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_objectives | - | 目标函数数量 |
| n_particles | 50 | 粒子数量（群体规模） |
| n_iterations | 100 | 最大迭代次数 |
| bounds | - | 决策变量边界 |
| w_initial | 0.9 | 惯性权重初始值 |
| w_final | 0.4 | 惯性权重终值 |
| G_initial | 100.0 | 引力因子初始值 |
| G_final | 1.0 | 引力因子终值 |
| archive_size | 粒子数量 | 外部档案容量，保持Pareto解集上限并通过拥挤距离选择 |

## 算法流程

1. **初始化群体**：随机生成粒子的位置和速度
2. **评估适应值**：计算每个粒子在各目标下的适应值
3. **聚集适应值**：使用随机权重方法计算聚合适应值
4. **计算质量**：根据适应值计算粒子质量
5. **计算力**：计算粒子间的虚拟引力和斥力
6. **更新速度和位置**：根据力和质量更新粒子运动状态
7. **更新Pareto前沿**：合并当前群体与外部档案，过滤非支配解并用拥挤距离截断至档案容量
8. **重复2-7**直到达到最大迭代次数

## 实验结果示例

运行快速测试的结果：

```
快速测试：Schaffer1函数
==================================================
Iteration 1/50, Pareto solutions: 3, w: 0.900, G: 100.000
Iteration 11/50, Pareto solutions: 30, w: 0.765, G: 80.200
Iteration 21/50, Pareto solutions: 30, w: 0.630, G: 60.400
Iteration 31/50, Pareto solutions: 30, w: 0.495, G: 40.600
Iteration 41/50, Pareto solutions: 30, w: 0.400, G: 20.800
Iteration 50/50, Pareto solutions: 30, w: 0.400, G: 2.980

性能指标:
  找到 30 个Pareto解
  GD: 0.003259
  SP: 0.069369
```

## 参考文献

王艳, 曾建潮. 一种基于拟态物理学优化的多目标优化算法[J]. 控制与决策, 2010, 25(7): 1040-1044.

```bibtex
@article{wang2010moapo,
  title={一种基于拟态物理学优化的多目标优化算法},
  author={王艳 and 曾建潮},
  journal={控制与决策},
  volume={25},
  number={7},
  pages={1040--1044},
  year={2010}
}
```

## 相关算法

- **APO** (Artificial Physics Optimization)：拟态物理学优化算法
- **PSO** (Particle Swarm Optimization)：粒子群优化算法
- **NSGA-II**：非支配排序遗传算法II
- **MOPSO**：多目标粒子群优化算法

## 使用建议

1. **粒子数量**：对于简单问题，30-50个粒子即可；复杂问题可增加到100个
2. **迭代次数**：一般100-200次迭代可以获得较好结果
3. **参数调优**：如果收敛过快，可以增大G_initial；如果收敛过慢，可以增大w_initial
4. **多次运行**：由于算法具有随机性，建议多次运行取最好结果

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进这个实现！

## 联系方式

如有问题或建议，请提交Issue。
