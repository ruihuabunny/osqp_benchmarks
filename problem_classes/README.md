# OSQP optimisation instance generators

本目录的 **7 类随机 optimisation instance generators 已完成复现**：实例生成、标准 QP 装配、OSQP 求解、CVXPY 模型与解恢复、固定 seed 复现以及参数更新均通过验证，共 **46/46 项检查通过**。

本文汇总复现报告和使用方式。**下一阶段的目标是尝试将这些 generators 接入主仓库的 [`src/generators/qp`](../../../src/generators/qp)**，作为项目中的 QP 实例来源。

## 1. 复现报告

### 1.1 范围与环境

本次复现覆盖下表中的随机生成器。每类使用 2 个尺寸和 `seed=0,1,2`，共 42 组基础实例；另外验证 4 类参数更新。

| 生成器 | 问题类型 | 验证尺寸 | 基础检查数 |
| --- | --- | --- | --- |
| `RandomQPExample` | 随机不等式约束 QP | `n=10,30` | 6 |
| `EqQPExample` | 等式约束 QP | `n=10,30` | 6 |
| `PortfolioExample` | 因子模型投资组合优化 | `k=3,5`，资产数默认 `100*k` | 6 |
| `LassoExample` | L1 正则化最小二乘 | 特征数 `n=5,10` | 6 |
| `HuberExample` | Huber 鲁棒回归 | 特征数 `n=5,10` | 6 |
| `SVMExample` | 软间隔支持向量机 | 特征数 `n=5,10` | 6 |
| `ControlExample` | 受约束最优控制 / MPC | `nx=4,10`，`nu=floor(nx/2)`，`T=10` | 6 |

已保存结果的记录时间为 **2026-09-07 12:11:34 UTC**，运行环境为 Linux/WSL2 x86_64、仓库根目录的 `.venv`。

| 依赖 | 复现时版本 |
| --- | --- |
| Python | 3.12.3 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| CVXPY | 1.9.2 |
| OSQP | 1.1.3 |

完整依赖快照见 [requirements-generators.lock](requirements-generators.lock)，原始测量见 [generator_validation_results.json](generator_validation_results.json)。更详细的过程与证据见 [GENERATORS_REPRODUCTION.md](GENERATORS_REPRODUCTION.md) 及其 [Notebook](GENERATORS_REPRODUCTION.ipynb)。

### 1.2 发现的问题与已完成的修复

修改前的 7 类代表实例均能生成并求解，但部分 CVXPY 表达式产生弃用警告，Lasso 的参数更新还存在两种模型不同步的问题。修改前测量保存在 [generator_validation_baseline.json](generator_validation_baseline.json)。

| 问题 | 修复与验证结果 |
| --- | --- |
| CVXPY 使用 `*` 表示矩阵/向量乘法 | 6 个生成器改用 `@`，相应求和改用 `cvxpy.sum()`；46 个验证用例均未捕获到警告 |
| Control 的 `cvxpy.vec()` 未指定展开顺序 | 显式使用 `order='F'`，与手工 QP 的时间排列保持一致 |
| Lasso 更新 lambda 后 CVXPY 目标函数仍引用旧常量 | 目标函数改为引用非负 `cvxpy.Parameter`；更新同步到 QP 与 CVXPY，负 lambda 在修改 QP 前被拒绝 |

例如，`LassoExample(10, seed=1)` 在 lambda 加倍后，修复前直接 QP 与 CVXPY 的目标值相差约 **8.23808**；修复后两者均约为 **1034.2135226400615**。验证还确认更新后的 QP 与依据当前属性重新装配的 QP 一致。

本轮保留了生成器构造接口、原问题结构、随机生成流程、稀疏 QP 输出、解恢复和参数更新方法。既有复现报告记录了 42/42 组初始 QP 数据在修复前后一致。

### 1.3 验证方法与结果

独立入口为 [validate_generators.py](validate_generators.py)。每个基础配置执行以下检查：

1. 构造实例并检查 `P,q,A,l,u,n,m` 的尺寸、稀疏矩阵、合法数值与上下界、Hessian 对称性及 CVXPY DCP 条件；有拆分边界字段时检查重组结果。
2. 使用标准 QP 直接调用 OSQP，再通过 CVXPY 调用 OSQP；恢复 CVXPY 原始/对偶解，并代回原 QP 检查约束、驻点、互补残差与目标值。
3. 用相同 seed 重建并检查数据一致；改用 `seed+100` 确认数据变化。对照构造不额外计入 42 个求解用例。
4. 对小规模 Control 检查动力学谱半径小于 1，以及初始状态位于状态边界内。

求解设置为 `eps_abs=eps_rel=1e-7`、`max_iter=100000`、`polishing=True`、`adaptive_rho_interval=50`、`verbose=False`，CVXPY 使用 `warm_start=False`。独立计算的归一化残差和目标差阈值为 `5e-6`，具体定义见验证脚本和详细报告。

| 检查类别 | 覆盖内容 | 结果 |
| --- | --- | --- |
| 基础实例 | 7 类 × 2 个尺寸 × 3 个 seed | 42/42 通过 |
| `lasso_lambda` | lambda 加倍；负值拒绝后状态不变 | 通过 |
| `control_x0` | 初始状态缩小至 0.5 倍 | 通过 |
| `portfolio_mu` | 更新收益向量，`k=3,n=60,seed=1` | 通过 |
| `portfolio_F_D` | F 乘 0.9、D 乘 1.1，保持稀疏结构 | 通过 |

全部 46 项的直接 OSQP 状态为 `solved`，CVXPY 状态为 `optimal`。保存记录中的主要误差如下；残差最大值包含两条求解路径。

| 指标 | 最大值 |
| --- | --- |
| 原始约束违反量（绝对值） | `2.887e-15` |
| 驻点残差（绝对值） | `5.800e-13` |
| 归一化原始残差 | `9.021e-16` |
| 归一化驻点残差 | `9.111e-14` |
| 归一化互补残差 | `2.605e-16` |
| 归一化两模型目标差 | `2.073e-14` |

记录中的 `run_suite()` 耗时约 **1.334 秒**，包含实例生成与验证，不含 Python 启动和安装。这是小规模正确性验证耗时，不是正式性能 benchmark。

### 1.4 结论范围

复现完成的范围是上述 7 类随机生成器及已测配置。`maros_meszaros.py`、`qplib.py`、`suitesparse_lasso.py`、`suitesparse_huber.py` 属于外部数据加载/转换入口，未纳入本次验证；相应的 `maros_meszaros_data/`、`qplib_data/`、`suitesparse_matrix_collection/` 数据也不是运行这些随机生成器的前提。

两条求解路径均使用 OSQP，本轮验证了模型转换和解恢复的一致性，未完成原论文全部 1400 个实例、多求解器性能比较或大规模扩容验证。

## 2. 使用方式

### 2.1 安装依赖与运行验证

以下终端命令适用于 Linux/WSL，均从**主仓库根目录**执行；已有 `.venv` 时跳过创建命令。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r third_parties/osqp_benchmarks/problem_classes/requirements-generators.lock
.venv/bin/python -m pip check
```

生成器依赖 NumPy、SciPy 和 CVXPY，直接求解使用 OSQP；锁文件还包含 Notebook 所需依赖。安装后，7 类随机实例生成与验证无需下载外部数据。

```bash
# 完整验证：42 组实例 + 4 类参数更新，成功时输出 46/46 passed
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/validate_generators.py --output /tmp/osqp-generators-rerun.json

# 快速验证：每类首个尺寸、seed=1，加上参数更新，共 11 项
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/validate_generators.py --quick
```

验证失败时脚本返回非零退出码。`--output` 可省略；上面的路径将重跑结果写入 `/tmp`，保留目录中原有的复现记录。

### 2.2 构造参数与实例规模

实例在**调用类的构造函数时生成**，同时创建 `qp_problem` 和 `cvxpy_problem`。构造函数不求解、不保存文件；直接运行 `python control.py` 等文件只会加载类定义。

| 文件与构造方式 | 参数含义 | 最终 QP 规模 `(n_QP, m_QP)` |
| --- | --- | --- |
| [random_qp.py](random_qp.py)：`RandomQPExample(n, seed=1)` | `n` 为原始变量数 | `(n, 10*n)` |
| [eq_qp.py](eq_qp.py)：`EqQPExample(n, seed=1)` | `n` 为原始变量数 | `(n, floor(n/2))` |
| [portfolio.py](portfolio.py)：`PortfolioExample(k, seed=1, n=None)` | `k` 为因子数；资产数 `n` 默认 `100*k` | `(n+k, n+k+1)` |
| [lasso.py](lasso.py)：`LassoExample(n, seed=1)` | `n` 为特征数，样本数为 `100*n` | `(102*n, 102*n)` |
| [huber.py](huber.py)：`HuberExample(n, seed=1)` | `n` 为特征数，样本数为 `100*n` | `(301*n, 300*n)` |
| [svm.py](svm.py)：`SVMExample(n, seed=1)` | `n` 为特征数，样本数为 `100*n` | `(101*n, 200*n)` |
| [control.py](control.py)：`ControlExample(n, seed=1)` | `nx=n`，`nu=floor(n/2)`，预测时域 `T=10` | `((T+1)*nx+T*nu, 2*(T+1)*nx+T*nu)` |

使用明确的正整数尺寸，Eq QP 和 Control 从 `n>=2` 开始。Portfolio 自定义资产数时使用 `PortfolioExample(3, seed=1, n=60)`，其第二个位置参数是 seed。Control 当前不支持通过构造参数独立指定 `nx/nu/T`。

### 2.3 生成实例并读取标准 QP

下列 Python 代码块按顺序执行，工作目录为主仓库根目录，解释器使用 `.venv/bin/python`；在 Notebook 中选择同一 `.venv` 内核。

```python
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "third_parties/osqp_benchmarks"))

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import osqp

from problem_classes.random_qp import RandomQPExample
from problem_classes.eq_qp import EqQPExample
from problem_classes.portfolio import PortfolioExample
from problem_classes.lasso import LassoExample
from problem_classes.huber import HuberExample
from problem_classes.svm import SVMExample
from problem_classes.control import ControlExample

instances = {
    "random_qp": RandomQPExample(10, seed=1),
    "eq_qp": EqQPExample(10, seed=1),
    "portfolio": PortfolioExample(5, seed=1),
    "lasso": LassoExample(10, seed=1),
    "huber": HuberExample(10, seed=1),
    "svm": SVMExample(10, seed=1),
    "control": ControlExample(10, seed=1),
}
for name, example in instances.items():
    qp = example.qp_problem
    print(name, "n_QP=", qp["n"], "m_QP=", qp["m"],
          "nnz(P)=", qp["P"].nnz, "nnz(A)=", qp["A"].nnz)
```

共同输出为标准凸 QP：

$$
\min_z\; \frac{1}{2}z^\mathsf{T}Pz + q^\mathsf{T}z,
\qquad l \leq Az \leq u.
$$

| `qp_problem` 字段 | 含义 |
| --- | --- |
| `P` | SciPy 稀疏矩阵，形状 `(n_QP, n_QP)`，完整对称 Hessian |
| `q` | 一维 NumPy 数组，形状 `(n_QP,)` |
| `A` | SciPy 稀疏 QP 约束矩阵，形状 `(m_QP, n_QP)` |
| `l`, `u` | 一维上下界数组，形状 `(m_QP,)`，可含合法无穷界 |
| `n`, `m` | 最终 QP 的变量数和约束行数，通常不同于构造参数 |

Control、Portfolio、Huber、SVM 还提供 `A_nobounds`、`l_nobounds`、`u_nobounds`、`bounds_idx`、`lx`、`ux`，用于拆分一般约束与变量边界。直接求解只需要 `P,q,A,l,u`，不要将整个字典展开给 `OSQP.setup()`。

**Control 的 `example.A` 是状态转移矩阵，`example.qp_problem['A']` 是 QP 约束矩阵。** Control 的 QP 变量按 `[vec_F(x), vec_F(u)]` 排列，即状态轨迹与控制轨迹分别按时间逐列展开。

### 2.4 求解与 CVXPY 交叉检查

```python
example = instances["lasso"]
qp = example.qp_problem
SETTINGS = dict(eps_abs=1e-7, eps_rel=1e-7, max_iter=100000,
                polishing=True, adaptive_rho_interval=50, verbose=False)

solver = osqp.OSQP()
solver.setup(**{key: qp[key] for key in ("P", "q", "A", "l", "u")}, **SETTINGS)
result = solver.solve(raise_error=True)
assert result.info.status == "solved", result.info.status
print("OSQP:", result.info.status, "objective:", result.info.obj_val)

example.cvxpy_problem.solve(solver=cp.OSQP, warm_start=False, **SETTINGS)
assert example.cvxpy_problem.status == cp.OPTIMAL
x_cvx, y_cvx = example.revert_cvxpy_solution()
assert x_cvx.shape == (qp["n"],) and y_cvx.shape == (qp["m"],)
np.testing.assert_allclose(result.info.obj_val, example.cvxpy_problem.value,
                           rtol=1e-6, atol=1e-6)
print("CVXPY:", example.cvxpy_problem.status, "objective:", example.cvxpy_problem.value)
```

该 Lasso 实例的目标值约为 `1025.9754409815434`。先成功求解 CVXPY 模型，再调用 `revert_cvxpy_solution()`，即可获得手工 QP 顺序下的原始解和对偶解。完整残差检查可运行第 2.1 节的验证命令。

### 2.5 保存与加载实例

以下示例保存上一步的 Lasso QP：`P/A` 使用稀疏 NPZ，向量、尺寸和 seed 使用 NumPy NPZ。输出位于 `/tmp/osqp-generator-example/lasso_n10_seed1`，重复运行会覆盖同名文件；需要长期保存时更换输出目录。

```python
output_dir = Path("/tmp/osqp-generator-example/lasso_n10_seed1")
output_dir.mkdir(parents=True, exist_ok=True)
sp.save_npz(output_dir / "P.npz", qp["P"].tocsc())
sp.save_npz(output_dir / "A.npz", qp["A"].tocsc())
np.savez_compressed(output_dir / "vectors.npz", q=qp["q"], l=qp["l"], u=qp["u"],
                    n=qp["n"], m=qp["m"], seed=1)

with np.load(output_dir / "vectors.npz", allow_pickle=False) as arrays:
    loaded = {key: arrays[key].copy() for key in ("q", "l", "u")}
    loaded.update(n=int(arrays["n"].item()), m=int(arrays["m"].item()))
loaded.update(P=sp.load_npz(output_dir / "P.npz"), A=sp.load_npz(output_dir / "A.npz"))

loaded_solver = osqp.OSQP()
loaded_solver.setup(**{key: loaded[key] for key in ("P", "q", "A", "l", "u")}, **SETTINGS)
loaded_result = loaded_solver.solve(raise_error=True)
assert loaded_result.info.status == "solved"
np.testing.assert_allclose(loaded_result.info.obj_val, result.info.obj_val,
                           rtol=1e-6, atol=1e-6)
print("已保存并重新加载求解:", output_dir)
```

该导出保存标准 QP 数据，不包含 CVXPY 对象、领域属性或拆分边界字段；加载后的五个标准字段足以重新求解。

### 2.6 参数更新

生成器的更新方法会同步其内部 QP/CVXPY 模型；用户已创建的 OSQP 对象需要另外更新。

| 方法 | 参数含义与要求 | 对已创建 OSQP 对象的操作 |
| --- | --- | --- |
| `lasso.update_lambda(lambda_new)` | 非负正则化系数 | `solver.update(q=lasso.qp_problem['q'])` |
| `control.update_x0(x0_new)` | 长度为 `nx` 的初始状态，需考虑边界和动力学可行性 | `solver.update(l=control.qp_problem['l'], u=control.qp_problem['u'])` |
| `portfolio.update_parameters(mu)` | 长度为资产数的收益向量 | `solver.update(q=portfolio.qp_problem['q'])` |
| `portfolio.update_parameters(mu, F=..., D=...)` | F/D 保持原尺寸与 CSC 稀疏结构，D 保持凸二次代价 | 用更新后的 QP 新建 OSQP 对象并 `setup()` |

继续前面的 Lasso 示例：

```python
example.update_lambda(2.0 * example.lambda_param)
solver.update(q=example.qp_problem["q"])
updated = solver.solve(raise_error=True)
assert updated.info.status == "solved"
example.cvxpy_problem.solve(solver=cp.OSQP, warm_start=False, **SETTINGS)
assert example.cvxpy_problem.status == cp.OPTIMAL
np.testing.assert_allclose(updated.info.obj_val, example.cvxpy_problem.value,
                           rtol=1e-6, atol=1e-6)
print("lambda 加倍后的目标值:", updated.info.obj_val)
```

更新后的目标值约为 `1034.2135226400615`。更多示例和 Notebook 启动方式见 [GENERATORS_USAGE.md](GENERATORS_USAGE.md) 及可执行的 [GENERATORS_USAGE.ipynb](GENERATORS_USAGE.ipynb)。

### 2.7 随机种子与规模

当前生成器使用 `np.random.seed(seed)`，会改变 NumPy 全局随机状态。固定依赖环境、构造参数与 seed 可复现实例；逐位一致性不延伸到任意依赖版本，也不应在线程中并发调用这些依赖全局 RNG 的构造函数。

扩大实例前应结合最终 QP 规模评估内存和生成成本：Lasso/Huber/SVM 的样本数固定为特征数的 100 倍；固定密度矩阵的非零数随尺寸增长，Random/Eq QP 的 Gram 矩阵还可能产生填充。Control 保留稠密特征分解和 Riccati 求解，稀疏 QP 装配本身不能消除这些瓶颈。更大尺寸与其他 seed 的可行性和数值表现需要继续验证。

## 3. 下一阶段目标：尝试接入 `src/generators/qp`

**下一阶段将尝试把本目录已复现的 7 类 generators 接入主仓库的 [`src/generators/qp`](../../../src/generators/qp)，使项目可以通过自身的生成入口获得这些问题族的标准 QP 实例。** 当前目标目录尚为空，具体接入接口有待实现。

建议按以下顺序推进：

1. 梳理主项目对 QP 实例的输入输出需求，明确问题族、尺寸参数和 seed 的传入方式，以现有 `P,q,A,l,u,n,m` 为数据基础。
2. 先选一类生成器打通从 `src/generators/qp` 调用、生成到下游消费的完整流程，再逐步接入其余 6 类；优先复用现有生成逻辑，保留各问题族的数学结构和参数含义。
3. 保持 `P/A` 稀疏存储、约束与变量排列、必要的边界信息；使用普通的问题族名称、参数和 seed 记录实例来源，明确依赖与导入方式。
4. 在相同环境、参数和 seed 下对照接入前后的 QP 数据、目标值及残差，复用现有 46 项验证，并检查主项目入口到下游消费的流程。
5. 在基本接入通过后评估批量生成和规模需求，结合实测结果再处理全局 RNG、固定维度比例和 Control 稠密计算等限制。

阶段验收目标是：主项目能选择这 7 类问题、指定已有尺寸参数与 seed、取得符合项目约定的稀疏 QP，并通过与当前复现基线的一致性及下游调用验证。
