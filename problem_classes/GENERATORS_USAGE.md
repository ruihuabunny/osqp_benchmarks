# optimisation instance generators 使用说明

本目录中的 7 类随机 optimisation instance generators 已完成实例生成、OSQP 求解和 CVXPY 模型一致性验证。依赖统一安装在**仓库根目录的 `.venv`**。实测环境和结果见 [复现报告](GENERATORS_REPRODUCTION.md)。

本文件对应 [GENERATORS_USAGE.ipynb](GENERATORS_USAGE.ipynb)。Notebook 保留本文说明，将 `python` 代码块转换为可执行单元；按顺序运行即可。`bash` 代码块是终端操作说明，不会在 Notebook 中自动安装依赖或启动服务。

## 1. 范围与入口

这些 `.py` 文件定义生成器类，**实例在调用构造函数时生成**。直接执行 `python control.py` 等文件只会加载类定义，不会生成、求解或保存实例。各构造函数同时创建 `qp_problem` 和 `cvxpy_problem`，本身不调用求解器。

| 文件 | 调用方式 | 首个参数含义 | 本次验证尺寸 |
| --- | --- | --- | --- |
| [random_qp.py](random_qp.py) | `RandomQPExample(n, seed=1)` | QP 变量数；约束数为 `10*n` | 10、30 |
| [eq_qp.py](eq_qp.py) | `EqQPExample(n, seed=1)` | QP 变量数；等式数为 `floor(n/2)` | 10、30 |
| [portfolio.py](portfolio.py) | `PortfolioExample(k, seed=1, n=None)` | 因子数 `k`；资产数 `n` 默认 `100*k` | k=3、5；另测 n=60 |
| [lasso.py](lasso.py) | `LassoExample(n, seed=1)` | 特征数；样本数 `100*n` | 5、10 |
| [huber.py](huber.py) | `HuberExample(n, seed=1)` | 特征数；样本数 `100*n` | 5、10 |
| [svm.py](svm.py) | `SVMExample(n, seed=1)` | 特征数；样本数 `100*n` | 5、10 |
| [control.py](control.py) | `ControlExample(n, seed=1)` | 状态数 `nx=n`；输入数 `nu=floor(n/2)` | 4、10 |

`seed` 默认均为 1。传入明确的正整数尺寸；Control 和 Eq QP 示例建议从 `n>=2` 开始。Portfolio 若指定资产数，应使用关键字 `n=...`，第二个位置参数是 `seed`。

以下三个文件夹不属于本次操作范围，未遍历、读取、下载或修改其内容：

- `maros_meszaros_data/`
- `qplib_data/`
- `suitesparse_matrix_collection/`

与它们配套的 `maros_meszaros.py`、`qplib.py`、`suitesparse_lasso.py`、`suitesparse_huber.py` 是外部数据加载/转换类，未实例化或验证。此次无需安装其额外的数据处理依赖。上级的 `run_benchmark_problems.py` 面向完整多求解器实验，涉及商业求解器和更多依赖；本说明使用生成器类和独立验证入口。

## 2. 在 `.venv` 安装和运行

以下终端命令适用于本次使用的 Linux/WSL，工作目录为仓库根目录。如果 `.venv` 已存在，跳过创建步骤。复现环境的 Python 版本为 3.12.3；完整版本固定在 [requirements-generators.lock](requirements-generators.lock)，包含实例生成、求解和 Notebook 所需的直接及传递依赖。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r third_parties/osqp_benchmarks/problem_classes/requirements-generators.lock
.venv/bin/python -m pip --no-cache-dir check
```

生成器本身需要 NumPy、SciPy 和 CVXPY；直接求解使用 OSQP。nbformat、nbclient、ipykernel 和 JupyterLab 用于 Notebook。无需激活环境，显式使用 `.venv/bin/python` 即可避免装到系统 Python；不要使用 `sudo pip` 或 `pip --user`。

运行全部 42 组实例和 4 类参数更新检查：

```bash
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/validate_generators.py --output /tmp/osqp-generators-rerun.json
```

成功时末行显示 `46/46 passed in ...s`，退出码为 0。`--quick` 只检查每类的首个尺寸、seed=1，加上 4 类参数更新，共 11 项：

```bash
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/validate_generators.py --quick
```

## 3. 准备 Python / Notebook 路径

本文 Python 代码按顺序执行。可从仓库根目录或其下级目录启动，代码会定位仓库根目录并检查内核使用 `.venv`。在普通 Python 脚本中也可使用同样的初始化代码。

```python
from pathlib import Path
import sys

ROOT = next(
    p for p in (Path.cwd(), *Path.cwd().parents)
    if (p / "third_parties/osqp_benchmarks/problem_classes/control.py").is_file()
)
assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "请切换到仓库 .venv 内核"
BENCHMARK_ROOT = ROOT / "third_parties/osqp_benchmarks"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

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

print("仓库:", ROOT)
print("解释器:", sys.executable)
```

## 4. 生成全部 7 类实例

```python
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
    data = example.qp_problem
    print(f"{name:12s} n_QP={data['n']:4d} m_QP={data['m']:4d} "
          f"nnz(P)={data['P'].nnz:4d} nnz(A)={data['A'].nnz:4d}")
```

共同输出是标准凸 QP：

$$\min_z\; \tfrac12 z^\mathsf{T}Pz+q^\mathsf{T}z,
\qquad l\leq Az\leq u.$$

| 字段 | 类型和含义 |
| --- | --- |
| `P` | SciPy 稀疏矩阵，shape=`(n_QP,n_QP)`，完整对称 Hessian |
| `q` | 一维 NumPy 数组，shape=`(n_QP,)` |
| `A` | SciPy 稀疏矩阵，shape=`(m_QP,n_QP)`，QP 约束矩阵 |
| `l`, `u` | 一维上下界数组，shape=`(m_QP,)`；允许合法的无穷界 |
| `n`, `m` | 最终 QP 的变量数和约束行数；通常不同于构造函数参数 |

Control、Portfolio、Huber、SVM 还提供 `A_nobounds`、`l_nobounds`、`u_nobounds`、`bounds_idx`、`lx`、`ux`，用于把变量边界从一般约束中分离。`lx/ux` 长度为 `n_QP`，`bounds_idx` 标识需要加入的变量边界行。直接使用 OSQP 时传入上述标准五项 `P,q,A,l,u` 即可，不能把整个 `qp_problem` 字典展开给 `setup()`。

**Control 的 `example.A` 是状态转移矩阵，`example.qp_problem['A']` 是 QP 约束矩阵，两者含义和尺寸不同。**

## 5. 尺寸、变量顺序与实际模型

下面用 `N` 表示生成器的特征数/资产数/原始变量数，`M` 表示样本数，避免与 `qp_problem['n']`、`['m']` 混淆。

| 类别 | QP 变量排列 | n_QP | m_QP | 实际目标与约束要点 |
| --- | --- | --- | --- | --- |
| Random QP | `x` | N | 10N | `P=G G.T+0.01 I`；构造可行见证后生成 `A x<=u` |
| Eq QP | `x` | N | floor(N/2) | 同类正定 Hessian；`A x=b` |
| Portfolio | `[x,y]` | N+k | N+k+1 | `x.T D x+y.T y-mu.T x/gamma`；`sum(x)=1, F.T x=y, 0<=x<=1` |
| Lasso | `[x,y,t]` | M+2N=102N | M+2N=102N | `||y||²+lambda*sum(t)`；`y=Ad x-bd, -t<=x<=t` |
| Huber | `[x,z,r,s]` | N+3M=301N | 3M=300N | `0.5||z||²+sum(r+s)`；`Ad x-bd-z=r-s, r,s>=0` |
| SVM | `[x,t]` | N+M=101N | 2M=200N | `0.5||x||²+0.5*gamma*sum(t)`；`t>=diag(b_svm) A_svm x+1, t>=0` |
| Control | `[vec_F(x),vec_F(u)]` | (T+1)nx+Tnu | 2(T+1)nx+Tnu | 状态/输入/终端二次代价、动力学、初始状态及上下界；T=10 |

Lasso 的残差平方项没有 `1/2`。SVM 的符号和 `1/2` 系数按当前源码列出。Huber 对应阈值 1 的损失：小残差为 `0.5*r²`，大残差为 `abs(r)-0.5`。Control 的状态和输入按时间逐列展开，使用 `order='F'`；当前终端代价保留源码中 Riccati 解与其转置的乘积。

## 6. 求解和交叉检查

以下以 Lasso 为例直接调用 `.venv` 中安装的 OSQP。`polishing` 是本次安装版本使用的参数名。

```python
SETTINGS = dict(eps_abs=1e-7, eps_rel=1e-7, max_iter=100000,
                polishing=True, adaptive_rho_interval=50, verbose=False)

example = instances["lasso"]
qp = example.qp_problem
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

该例的目标值约为 `1025.9754409815434`。`revert_cvxpy_solution()` 将 CVXPY 的原始和对偶解映射回手工 QP 的变量/约束顺序；先成功求解再调用。两条路径都使用 OSQP，主要验证模型转换及解映射，不构成不同求解器之间的独立比较。完整的残差检查在 [validate_generators.py](validate_generators.py) 中。

## 7. 保存与重新加载稀疏 QP

构造函数不会自动写文件。示例把矩阵分别存为稀疏 NPZ，向量和尺寸存为另一份 NPZ，无需 pickle。示例产物放在 `.venv/osqp-generator-artifacts/`，重复运行覆盖同名示例文件；正式数据可另选输出路径。

```python
output_dir = ROOT / ".venv/osqp-generator-artifacts/lasso_n10_seed1"
output_dir.mkdir(parents=True, exist_ok=True)
sp.save_npz(output_dir / "P.npz", qp["P"].tocsc())
sp.save_npz(output_dir / "A.npz", qp["A"].tocsc())
np.savez_compressed(output_dir / "vectors.npz", q=qp["q"], l=qp["l"], u=qp["u"],
                    n=qp["n"], m=qp["m"], leading_dimension=10, seed=1)

with np.load(output_dir / "vectors.npz", allow_pickle=False) as arrays:
    loaded = {key: arrays[key].copy() for key in ("q", "l", "u")}
    loaded.update(n=int(arrays["n"].item()), m=int(arrays["m"].item()))
loaded.update(P=sp.load_npz(output_dir / "P.npz"), A=sp.load_npz(output_dir / "A.npz"))

from problem_classes.validate_generators import qp_fingerprint
standard_fields = {key: qp[key] for key in ("P", "q", "A", "l", "u", "n", "m")}
assert qp_fingerprint(loaded) == qp_fingerprint(standard_fields)
print("保存/加载一致:", output_dir)
```

这份导出只保存求解所需的标准 QP，不保存 CVXPY 对象、领域属性或单独拆分的变量边界。重新求解可将 `loaded` 的五个标准字段传给 `OSQP.setup()`。

## 8. 参数更新

| 方法 | 含义 | 已创建 OSQP 对象需要的动作 |
| --- | --- | --- |
| `lasso.update_lambda(lambda_new)` | 更新非负正则化系数及两种模型；负值抛出 `ValueError` | `solver.update(q=...)` |
| `control.update_x0(x0_new)` | 更新长度为 `nx` 的初始状态、QP 等式边界和 CVXPY 参数 | `solver.update(l=..., u=...)` |
| `portfolio.update_parameters(mu)` | 更新收益向量及两种模型 | `solver.update(q=...)` |
| `portfolio.update_parameters(mu, F=..., D=...)` | F、D 必须保持 CSC 稀疏结构；D 应保持适当的凸二次代价 | 简单可靠的方式是重新 `setup()` |

生成器方法不会自动更新用户已创建的 OSQP 对象。Lasso 的完整更新示例：

```python
example.update_lambda(2.0 * example.lambda_param)
solver.update(q=example.qp_problem["q"])
updated = solver.solve(raise_error=True)
assert updated.info.status == "solved"
example.cvxpy_problem.solve(solver=cp.OSQP, warm_start=False, **SETTINGS)
assert example.cvxpy_problem.status == cp.OPTIMAL
np.testing.assert_allclose(updated.info.obj_val, example.cvxpy_problem.value,
                           rtol=1e-6, atol=1e-6)
print("更新后目标值:", updated.info.obj_val)

control = instances["control"]
control.update_x0(0.5 * control.x0)
portfolio = instances["portfolio"]
portfolio.update_parameters(portfolio.mu + np.linspace(0, 0.2, portfolio.n))
print("Control x0 和 Portfolio mu 已同步到生成器的两种模型")
```

Lasso 更新后目标值约为 `1034.2135226400615`。Control 的新初始状态需由调用者考虑边界及后续动力学可行性，`update_x0()` 不保证任意输入均可行。

## 9. 随机种子和规模限制

当前生成器使用 `np.random.seed(seed)`，会改变 NumPy 全局随机状态。同一固定依赖环境、尺寸和 seed 的标准 QP 可以复现；本次每个测试配置都验证了重复生成一致、换用 `seed+100` 后数据不同。不应假设跨 NumPy/SciPy 版本仍逐位一致，也不宜在线程中并发调用这些依赖全局 RNG 的构造函数。

本次验证的是表中小规模配置，尚未验证原始论文的全部 1400 个实例或超大规模性能。放大尺寸前需考虑：

- Lasso/Huber/SVM 的样本数固定为特征数的 100 倍；Huber 的最终变量数达到 `301*n`。
- 固定密度的随机矩阵非零数仍随维度近似平方增长；Random/Eq QP 的 `G @ G.T` 可能产生填充。
- Control 当前 `A/B` 为稠密数据的稀疏存储，包含稠密特征分解和 Riccati 方程求解；其 `T=10` 固定，构造函数暂不接收 `nx/nu/T` 独立参数。稀疏 QP 装配不代表前期生成没有稠密瓶颈。
- 本次测试的 Control 实例可行，不能据此保证任意尺寸和 seed 下都可行。增大尺寸后应继续检查 `status` 和残差。

## 10. 打开 Notebook

将内核注册到 `.venv` 本身，再用同一环境启动 JupyterLab：

```bash
IPYTHONDIR=.venv/ipython .venv/bin/python -m ipykernel install --sys-prefix --name osqp-generators --display-name "Python (.venv - OSQP generators)"
IPYTHONDIR=.venv/ipython JUPYTER_CONFIG_DIR=.venv/jupyter/config JUPYTER_RUNTIME_DIR=.venv/jupyter/runtime .venv/bin/jupyter lab --no-browser
```

在浏览器打开终端给出的本地地址，选择本文 Notebook，并选择 `Python (.venv - OSQP generators)` 内核。在 VS Code 中也可直接选择仓库 `.venv/bin/python`。移到另一台机器或另一条仓库路径后，重新执行内核注册命令。

最后可以在当前内核运行快速验证：

```python
from problem_classes.validate_generators import run_suite
quick_result = run_suite(quick=True)
assert quick_result["all_passed"]
assert quick_result["total"] == 11
print(f"{quick_result['passed']}/{quick_result['total']} passed")
```
