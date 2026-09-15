# optimisation instance generators 使用说明

本目录中的 7 类随机 optimisation instance generators 已完成实例生成、OSQP 求解和 CVXPY 模型一致性验证。依赖统一安装在**仓库根目录的 `.venv`**。历史实测环境和结果见 [复现报告](GENERATORS_REPRODUCTION.md)；本文按当前源码说明接口，其中 Random QP、Control、Eq QP、Lasso 和 Huber 已支持自定义生成参数。

本文件对应 [GENERATORS_USAGE.ipynb](GENERATORS_USAGE.ipynb)。Notebook 保留本文说明，将 `python` 代码块转换为可执行单元；按顺序运行即可。`bash` 代码块是终端操作说明，不会在 Notebook 中自动安装依赖或启动服务。

## 1. 范围与入口

这些 `.py` 文件定义生成器类，**实例在调用构造函数时生成**。直接执行 `python control.py` 等文件只会加载类定义，不会生成、求解或保存实例。各构造函数同时创建 `qp_problem` 和 `cvxpy_problem`，本身不调用求解器。

| 文件 | 调用方式 | 尺寸参数含义 | 本次验证尺寸 |
| --- | --- | --- | --- |
| [random_qp.py](random_qp.py) | `RandomQPExample(n, m, P_block_sizes, P_rank, P_lambda_max, P_cond_num, A_density, q_scale, slack_scale, seed=1)` | `n`、`m` 独立控制 QP 变量数和不等式约束数 | 10、30 |
| [eq_qp.py](eq_qp.py) | `EqQPExample(n, m=None, lower_density=0.15, seed=1, ...)` | QP 变量数；等式数 `m` 默认 `floor(n/2)` | 10、30 |
| [portfolio.py](portfolio.py) | `PortfolioExample(k, seed=1, n=None)` | 因子数 `k`；资产数 `n` 默认 `100*k` | k=3、5；另测 n=60 |
| [lasso.py](lasso.py) | `LassoExample(n, seed=1, m=None, ...)` | 特征数；样本数 `m` 默认 `100*n` | 5、10 |
| [huber.py](huber.py) | `HuberExample(n, m, num_blocks, block_sizes, r, max_singular_val, max_cond_num, delta, sigma, outlier_fraction, outlier_scale, seed=1)` | `n` 为特征数，`m` 为样本数；还需指定分块、谱和噪声参数 | `(n,m)=(5,500)、(10,1000)`；另测分块示例 `(20,100)` |
| [svm.py](svm.py) | `SVMExample(n, seed=1)` | 特征数；样本数 `100*n` | 5、10 |
| [control.py](control.py) | `ControlExample(nx, nu, seed=1)` | `nx` 为状态数，`nu` 为输入数，均必填且可独立设置 | `(nx,nu)=(4,2)、(10,5)`；参数测试另含 `(1,2)、(10,8)、(4,4)、(4,6)` |

`seed` 默认均为 1。传入明确的正整数尺寸；Control 使用 `nx>=1, nu>=1`，Eq QP 使用默认 `m` 时从 `n>=2` 开始。Portfolio 若指定资产数，应使用关键字 `n=...`，第二个位置参数是 `seed`。

Control 的第二个位置参数现在是 `nu`，Random QP、Eq QP 和 Huber 的第二个位置参数是 `m`，请通过关键字指定 `seed`；Lasso 的第二个位置参数仍是 `seed`。原来的 `ControlExample(10, seed=1)` 应改为 `ControlExample(nx=10, nu=5, seed=1)`；旧的 `HuberExample(n, seed=...)` 也需要补齐参数，见第 4、5.9 节。

**五个生成器的当前实现与可控项：**

| 文件 | 生成流程 | 变量数与约束数控制 | 矩阵和数据控制 |
| --- | --- | --- | --- |
| `random_qp.py` | 块内正交变换构造半正定 P，再生成全局稀疏 A 和可行上界 | `n`、`m` 独立控制最终 QP 尺寸 | `P_block_sizes/P_rank` 控制分块与秩；`P_lambda_max/P_cond_num` 控制谱上界；`A_density/q_scale/slack_scale` 控制约束密度和数据尺度 |
| `control.py` | 生成有限时域 MPC 的动力学、代价和边界，再组装稀疏 QP；同时建立 CVXPY MPC 模型 | `nx`、`nu` 独立控制状态数和输入数；`T=10`，最终 QP 尺寸由轨迹展开决定 | 当前构造参数为 `nx, nu, seed`；动力学和代价矩阵按内部规则生成 |
| `eq_qp.py` | 直接生成等式约束凸 QP | `n` 为 QP 变量数；`m` 为等式约束行数，可独立设置且 `1<=m<=n` | `lower_density` 控制构造矩阵和约束矩阵的非零密度；`r` 控制 Hessian 秩；`max_spectrum`、`max_cond_num` 控制 Hessian 的谱和条件数上界 |
| `lasso.py` | 生成回归数据和 L1 正则化最小二乘问题，再引入残差与辅助变量转为 QP | `n` 为特征数，`m` 为样本数，可独立设置；最终 QP 变量数和约束行数均为 `m+2*n` | `density` 控制数据矩阵非零密度，`data_scale` 控制数据尺度，`lambda_ratio` 控制 L1 正则化强度 |
| `huber.py` | 按指定块尺寸和奇异值生成块对角回归矩阵，再将 Huber 损失转为凸 QP | `n`、`m` 独立控制特征数和样本数；最终 QP 为 `n+3*m` 个变量、`3*m` 行约束 | `num_blocks/block_sizes/r` 控制分块与秩；`max_singular_val/max_cond_num` 控制正奇异值；`delta` 控制损失阈值，`sigma/outlier_fraction/outlier_scale` 控制噪声 |

本文的 `density` / `lower_density` 表示**非零元素比例**；例如 `0.05` 表示约 5% 非零、95% 为零。具体作用的矩阵、默认值、限制和可执行示例见第 5.1–5.6 节（Control）、第 5.7 节（Eq QP）、第 5.8 节（Lasso）和第 5.9 节（Huber）。

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
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/tests/test_generators.py --output /tmp/osqp-generators-rerun.json
```

成功时末行显示 `46/46 passed in ...s`，退出码为 0。`--quick` 只检查每类的首个尺寸、seed=1，加上 4 类参数更新，共 11 项：

```bash
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/tests/test_generators.py --quick
```

Control、Eq QP 和 Lasso 的参数测试覆盖尺寸、密度、秩与谱上界、数据缩放、正则化阈值、参数更新和可复现性。Control 包含 6 组 `(nx,nu)`、每组 3 个 seed 的生成与求解检查，以及独立的复现和 `update_x0()` 检查：

```bash
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/tests/test_generator_parameters.py
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/tests/test_random_qp_parameters.py
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
    "random_qp": RandomQPExample(
        n=10, m=100, P_block_sizes=[5, 5], P_rank=10,
        P_lambda_max=10.0, P_cond_num=100.0, A_density=0.15,
        q_scale=1.0, slack_scale=1.0, seed=1),
    "eq_qp": EqQPExample(10, seed=1),
    "portfolio": PortfolioExample(5, seed=1),
    "lasso": LassoExample(10, seed=1),
    "huber": HuberExample(
        n=20, m=100, num_blocks=2, block_sizes=[(60, 12, 8), (40, 8, 4)],
        r=12, max_singular_val=10.0, max_cond_num=100.0, delta=1.5,
        sigma=0.5, outlier_fraction=0.05, outlier_scale=10.0, seed=1,
    ),
    "svm": SVMExample(10, seed=1),
    "control": ControlExample(nx=10, nu=5, seed=1),
}
for name, example in instances.items():
    data = example.qp_problem
    print(f"{name:12s} n_QP={data['n']:4d} m_QP={data['m']:4d} "
          f"nnz(P)={data['P'].nnz:4d} nnz(A)={data['A'].nnz:4d}")
```

Random QP 的 `P_block_sizes` 是总和等于 `n` 的正整数序列，`1 <= P_rank <= n`。正特征值在 `[P_lambda_max/P_cond_num, P_lambda_max]` 均匀抽样，混入零后打乱并分配给各块。有限正数 `P_lambda_max` 和有限的 `P_cond_num >= 1` 均为上界，不要求取到；后者控制非零谱条件数，秩亏时普通条件数为无穷大。

`A_density` 取 `[0,1]`，控制全局 `m×n` 标准高斯稀疏矩阵，约束可以连接不同的 P 块。有限的 `q_scale >= 0` 缩放正特征子空间中的高斯向量，保证目标有下界；满秩时 q 为缩放的标准高斯向量。有限的 `slack_scale > 0` 控制保存的可行点 `v` 处的代数松弛 `u-A@v`，不代表欧氏距离、最优解或最优解处活跃约束数。

`P_eigenvalues` 保存生成的谱，`P_lambda_max_actual` 和 `P_cond_num_actual` 直接由正特征值计算。`P_density_actual`、`A_density_actual` 按最终 CSC 矩阵清除显式零后的 `nnz` 统计；`P_sparsity_actual`、`A_sparsity_actual` 为对应的 `1-density`。构造过程中只在块内做稠密运算。

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
| Random QP | `x` | N | m | 具有指定秩和谱上界的半正定块对角 P；`q` 位于 `range(P)`；`u=A@v+slack_scale*delta`，`l=-inf` |
| Eq QP | `x` | N | m（默认 floor(N/2)） | 可控秩的半正定 Hessian；`A x=b` |
| Portfolio | `[x,y]` | N+k | N+k+1 | `x.T D x+y.T y-mu.T x/gamma`；`sum(x)=1, F.T x=y, 0<=x<=1` |
| Lasso | `[x,y,t]` | M+2N（默认 102N） | M+2N（默认 102N） | `||y||²+lambda*sum(t)`；`y=Ad x-bd, -t<=x<=t` |
| Huber | `[x,z,r,s]` | N+3M | 3M | `0.5||z||²+delta*sum(r+s)`；`Ad x-bd-z=r-s, r,s>=0` |
| SVM | `[x,t]` | N+M=101N | 2M=200N | `0.5||x||²+0.5*gamma*sum(t)`；`t>=diag(b_svm) A_svm x+1, t>=0` |
| Control | `[vec_F(x),vec_F(u)]` | (T+1)nx+Tnu | 2(T+1)nx+Tnu | 状态/输入/终端二次代价、动力学、初始状态及上下界；T=10 |

Lasso 的残差平方项没有 `1/2`。SVM 的符号和 `1/2` 系数按当前源码列出。Huber 对应阈值 `delta` 的损失：当 `abs(t)<=delta` 时为 `0.5*t²`，否则为 `delta*abs(t)-0.5*delta²`。Control 的状态和输入按时间逐列展开，使用 `order='F'`；当前终端代价保留源码中 Riccati 解与其转置的乘积。

### 5.1 Control：`__init__` 参数与 MPC 模型

当前调用接口为 `ControlExample(nx, nu, seed=1)`，`nx`、`nu` 均为必填的正整数。`ControlExample.__init__()` 先生成有限时域 MPC 的参数，`_generate_qp_problem()` 再读取这些 `self` 属性，将整个预测区间的目标和约束组装成 QP；`_generate_cvxpy_problem()` 同时建立对应的 MPC 模型。构造函数完成建模，求解需另行调用 OSQP 或 CVXPY。

生成顺序为：`nx, nu, seed` → 状态/输入维数 → 动力学 `A, B` → 代价 `Q, R, QN` → 状态/输入边界与 `x0` → 固定预测步数 `T` → QP 和 CVXPY 模型。当前实现使用：

$$
n_x=\texttt{self.nx}=\texttt{nx},\qquad
n_u=\texttt{self.nu}=\texttt{nu},\qquad
T=\texttt{self.T}=10.
$$

`nx` 和 `nu` 可独立设置，没有 `2*nu<=nx` 或 `nu<=nx` 的要求；`T` 仍固定为 10。最终 QP 的变量数和约束数按第 5.2、5.4 节的公式变化，不能把构造参数当成最终 QP 的 `n,m`。矩阵密度、谱、条件数和预测步数尚未作为构造参数开放。`update_x0()` 可在生成后同步更新两种模型的初始状态，见第 8 节。

下面用 $A_d$ 表示 `self.A`，用 $A_{\mathrm{QP}}$ 表示 `qp_problem['A']`；用 $x_{\mathrm{init}}$ 表示已知参数 `self.x0`。向量的 shape 按 NumPy 一维数组写成 `(长度,)`。

| `__init__` 属性 | shape | 在 MPC 中的作用 |
| --- | --- | --- |
| `self.A`，记为 $A_d$ | $(n_x,n_x)$ | 状态转移矩阵；决定当前状态如何影响下一步状态 |
| `self.B` | $(n_x,n_u)$ | 控制输入矩阵；决定各控制输入如何影响状态 |
| `self.Q` | $(n_x,n_x)$ | 过程中的状态代价矩阵，当前为非负对角矩阵 |
| `self.R` | $(n_u,n_u)$ | 控制输入代价矩阵，当前为 $0.1I_{n_u}$ |
| `self.QN`，记为 $Q_N$ | $(n_x,n_x)$ | 预测最后一步的终端状态代价矩阵 |
| `self.x0` | $(n_x,)$ | 已知初始状态 |
| `self.xmin`、`self.xmax` | $(n_x,)$ | 每一步状态的逐分量下界和上界 |
| `self.umin`、`self.umax` | $(n_u,)$ | 每一步控制输入的逐分量下界和上界 |
| `self.nx`、`self.nu`、`self.T` | 标量 | 单步状态维数、单步输入维数、预测步数 |

这些属性定义如下有限时域 MPC 问题：

$$
\begin{aligned}
\min_{x_0,\ldots,x_T,\,u_0,\ldots,u_{T-1}}\quad
&\sum_{k=0}^{T-1}\left(x_k^\mathsf{T}Qx_k+u_k^\mathsf{T}Ru_k\right)
 +x_T^\mathsf{T}Q_Nx_T\\
\text{s.t.}\quad
&x_0=x_{\mathrm{init}},\\
&x_{k+1}=A_dx_k+Bu_k, &&k=0,\ldots,T-1,\\
&x_{\min}\leq x_k\leq x_{\max}, &&k=0,\ldots,T,\\
&u_{\min}\leq u_k\leq u_{\max}, &&k=0,\ldots,T-1.
\end{aligned}
$$

$A_d$、$B$ 决定状态如何演化；$Q$、$R$、$Q_N$ 决定如何评价一条轨迹。相对其他代价增大 $R$，会更重视节省控制用量；$Q_N$ 则使优化兼顾预测终点的状态偏差。这里的状态代价以零状态为参照。

源码会调整状态转移矩阵 `self.A` 的特征值，使其模小于 1，以控制离散动力学的稳定性。

终端代价的实际生成过程为：

$$
S=\operatorname{solve\_discrete\_are}(A_d,B,Q,R),\qquad
Q_N=SS^\mathsf{T}\succeq0.
$$

因此当前使用的终端权重是 Riccati 解与其转置的乘积；它与直接使用 Riccati 解 $S$ 的终端代价不同。

### 5.2 Control：轨迹变量展开与 QP 变量数

MPC 的状态轨迹和控制轨迹为：

$$
X=[x_0,\ldots,x_T]\in\mathbb R^{n_x\times(T+1)},\qquad
U=[u_0,\ldots,u_{T-1}]\in\mathbb R^{n_u\times T}.
$$

这对应 `_generate_cvxpy_problem()` 中 shape 为 `(nx, T+1)` 的 `x` 和 shape 为 `(nu, T)` 的 `u`。QP 使用按时间逐列展开的单个向量：

$$
z=
\begin{bmatrix}
\operatorname{vec}_F(X)\\
\operatorname{vec}_F(U)
\end{bmatrix}
=
\begin{bmatrix}
x_0\\\vdots\\x_T\\u_0\\\vdots\\u_{T-1}
\end{bmatrix}.
$$

定义状态变量总数和控制变量总数：

$$
s=(T+1)n_x,\qquad c=Tn_u.
$$

于是：

$$
\boxed{n_{\mathrm{QP}}=s+c=(T+1)n_x+Tn_u},\qquad
z\in\mathbb R^{n_{\mathrm{QP}}}.
$$

`x_0` 也占用 $n_x$ 个决策变量，通过初始状态等式固定为 `self.x0`。`qp_problem` 保存目标与约束的系数；求解器创建并求解 $z$，其 shape 为 `(n_QP,)`。

### 5.3 Control：二次目标的块矩阵与 shape

OSQP 的目标形式为：

$$
\min_z\;\frac12z^\mathsf{T}Pz+q^\mathsf{T}z.
$$

代码用 Kronecker 积和块对角拼接构造：

$$
P_x=I_T\otimes Q\in\mathbb R^{Tn_x\times Tn_x},\qquad
P_u=I_T\otimes R\in\mathbb R^{c\times c},
$$

$$
\begin{aligned}
P
&=2\operatorname{blkdiag}(P_x,Q_N,P_u)\\
&=2\operatorname{blkdiag}
\left(\underbrace{Q,\ldots,Q}_{T\text{ 个}},Q_N,
\underbrace{R,\ldots,R}_{T\text{ 个}}\right)
\in\mathbb R^{n_{\mathrm{QP}}\times n_{\mathrm{QP}}},\\
q&=0\in\mathbb R^{n_{\mathrm{QP}}}.
\end{aligned}
$$

`spa.kron(I_T, Q)` 在对角线上重复放置 $T$ 个 $Q$。乘以 2 抵消 OSQP 目标前的 $1/2$；MPC 目标没有一次项，因此 `q` 全为零。三个块的总维度为 $Tn_x+n_x+c=s+c$，与 $z$ 的排列一致。

### 5.4 Control：约束组装、行数与 shape

初始条件和动力学先写成：

$$
-x_0=-x_{\mathrm{init}},\qquad
A_dx_k-x_{k+1}+Bu_k=0,\quad k=0,\ldots,T-1.
$$

设 $J\in\mathbb R^{(T+1)\times(T+1)}$ 的第一条下副对角线全为 1，其余为 0，则代码中的状态和输入系数块为：

$$
A_x=-I_s+J\otimes A_d\in\mathbb R^{s\times s},\qquad
A_u=
\begin{bmatrix}
0_{1\times T}\\I_T
\end{bmatrix}\otimes B
\in\mathbb R^{s\times c}.
$$

横向拼接得到：

$$
D=[A_x\;A_u]\in\mathbb R^{s\times n_{\mathrm{QP}}},\qquad
b=
\begin{bmatrix}
-x_{\mathrm{init}}\\0_{Tn_x}
\end{bmatrix}\in\mathbb R^s,
\qquad Dz=b.
$$

例如只为展示结构而取 $T=2$ 时：

$$
\begin{bmatrix}
-I&0&0&0&0\\
A_d&-I&0&B&0\\
0&A_d&-I&0&B
\end{bmatrix}
\begin{bmatrix}
x_0\\x_1\\x_2\\u_0\\u_1
\end{bmatrix}
=
\begin{bmatrix}
-x_{\mathrm{init}}\\0\\0
\end{bmatrix}.
$$

初始条件贡献 $n_x$ 行，$T$ 步动力学贡献 $Tn_x$ 行，共 $s$ 行。函数先保存 `A_nobounds = D`、`l_nobounds = u_nobounds = b`。等式通过相同上下界表示：$b\leq Dz\leq b$。

随后追加状态上下界和输入上下界，得到最终约束矩阵：

$$
A_{\mathrm{QP}}=
\begin{bmatrix}
A_x&A_u\\
I_s&0_{s\times c}\\
0_{c\times s}&I_c
\end{bmatrix}
\in\mathbb R^{(2s+c)\times(s+c)}.
$$

对应的上下界为：

$$
\ell=
\begin{bmatrix}
b\\
\mathbf1_{T+1}\otimes x_{\min}\\
\mathbf1_T\otimes u_{\min}
\end{bmatrix},\qquad
u_{\mathrm{QP}}=
\begin{bmatrix}
b\\
\mathbf1_{T+1}\otimes x_{\max}\\
\mathbf1_T\otimes u_{\max}
\end{bmatrix}
\in\mathbb R^{2s+c}.
$$

$\mathbf1_r\otimes v$ 表示将向量 $v$ 重复堆叠 $r$ 次，对应 `np.tile(v, r)`。这里 $u_{\mathrm{QP}}$ 是 `qp_problem['u']` 的约束上界向量；控制输入是 $u_k$，其轨迹存放在 $z$ 的后 $c$ 个分量中。

| 约束类型 | 标量约束行数 | 在最终 `A` 中的行切片（从 0 开始） |
| --- | --- | --- |
| 初始状态 $x_0=x_{\mathrm{init}}$ | $n_x$ | `0:nx` |
| $T$ 步动力学 | $Tn_x$ | `nx:s` |
| 所有状态的上下界 | $s=(T+1)n_x$ | `s:2*s` |
| 所有控制输入的上下界 | $c=Tn_u$ | `2*s:2*s+c` |

每个标量变量的下界和上界合并为一行双侧约束，因此变量上下界一共贡献 $s+c$ 行。最终：

$$
\boxed{m_{\mathrm{QP}}=2s+c=2(T+1)n_x+Tn_u},\qquad
\ell\leq A_{\mathrm{QP}}z\leq u_{\mathrm{QP}}.
$$

### 5.5 Control：`qp_problem` 输出 shape 汇总

| 字段 | shape 或值 | 对应内容 |
| --- | --- | --- |
| `P` | `(n_QP, n_QP)` | 状态、终端状态及控制输入的二次代价 |
| `q` | `(n_QP,)` | 全零一次项 |
| `A` | `(m_QP, n_QP)` | 初始条件、动力学、状态和输入上下界 |
| `l`、`u` | `(m_QP,)` | 全部约束的下界、上界 |
| `n` | 标量，值为 $n_{\mathrm{QP}}=s+c$ | QP 决策变量总数 |
| `m` | 标量，值为 $m_{\mathrm{QP}}=2s+c$ | QP 约束行数 |
| `A_nobounds` | `(s, n_QP)` | 初始条件和动力学的系数矩阵 $D$ |
| `l_nobounds`、`u_nobounds` | `(s,)` | 均等于等式右端向量 $b$ |
| `lx`、`ux` | `(n_QP,)` | 按 $z$ 的顺序排列的状态和控制输入上下界 |
| `bounds_idx` | `(n_QP,)` | 所有变量的索引 `0, ..., n_QP-1` |

`lx/ux` 包括全部状态和控制输入的变量边界；它们相当于从 `l/u` 中去掉前 $s$ 个等式边界。`P` 和 `A` 的 shape 是逻辑矩阵维度，实际使用 SciPy 稀疏矩阵存储。求解得到的原始变量 $z$ 的 shape 为 `(n_QP,)`，对偶变量 $y$ 的 shape 为 `(m_QP,)`；它们不存放在 `qp_problem` 字典中。

### 5.6 Control：具体尺寸与可运行示例

例如显式指定 `nx=100, nu=50` 时，$n_x=100$、$n_u=50$、$T=10$，所以：

$$
s=11\times100=1100,\qquad c=10\times50=500,
$$

$$
\boxed{n_{\mathrm{QP}}=1600,\qquad m_{\mathrm{QP}}=2700}.
$$

| 对象 | `nx=100, nu=50` 时的 shape |
| --- | --- |
| `self.A`、`self.Q`、`self.QN` | `(100, 100)` |
| `self.B` | `(100, 50)` |
| `self.R` | `(50, 50)` |
| 状态轨迹 $X$、控制轨迹 $U$ | `(100, 11)`、`(50, 10)` |
| QP 决策向量 $z$ | `(1600,)` |
| `P`、`q` | `(1600, 1600)`、`(1600,)` |
| `A` | `(2700, 1600)` |
| `l`、`u` | `(2700,)` |
| `A_nobounds` | `(1100, 1600)` |
| `l_nobounds`、`u_nobounds` | `(1100,)` |
| `lx`、`ux`、`bounds_idx` | `(1600,)` |

这个表按维度公式计算，表示构造后的逻辑尺寸，不表示该规模已经通过求解或性能验证。构造函数输入的 `nx=100` 是单步状态维数，`nu=50` 是单步输入维数，输出 `qp_problem['n']=1600` 是整条预测轨迹的决策变量总数。

下面复用第 4 节已生成的 `ControlExample(nx=10, nu=5, seed=1)`，打印实际 shape；此时应得到 `n_QP=160`、`m_QP=270`。代码使用独立的 `control_example`、`control_qp` 名称，便于后面的 Lasso 示例继续执行。

```python
control_example = instances["control"]
control_qp = control_example.qp_problem
nx, nu, horizon = control_example.nx, control_example.nu, control_example.T
state_count = (horizon + 1) * nx
input_count = horizon * nu

print(f"nx={nx}, nu={nu}, T={horizon}")
print("公式计算:", f"n_QP={state_count + input_count}, "
      f"m_QP={2 * state_count + input_count}")
print("实际输出:", f"n_QP={control_qp['n']}, m_QP={control_qp['m']}")
for name in ("A", "B", "Q", "R", "QN", "x0", "xmin", "xmax", "umin", "umax"):
    print(f"self.{name:5s}: {getattr(control_example, name).shape}")

state_trajectory, input_trajectory = control_example.cvxpy_variables
print("状态轨迹 X:", state_trajectory.shape)
print("控制轨迹 U:", input_trajectory.shape)
for key in ("P", "q", "A", "l", "u", "A_nobounds", "l_nobounds",
            "u_nobounds", "lx", "ux", "bounds_idx"):
    print(f"qp_problem[{key!r}]: {control_qp[key].shape}")
```

独立设置 `nu` 的可运行示例如下。`(nx,nu)=(10,8)` 满足 `2*nu>nx`，`(4,6)` 还满足 `nu>nx`。检查函数会分别用 OSQP 和 CVXPY+OSQP 求解，比较目标值，并检查原始/对偶解的残差与映射。

```python
from problem_classes.tests.test_generators import solve_and_check

for nx, nu in ((10, 8), (4, 6)):
    mpc_example = ControlExample(nx=nx, nu=nu, seed=1)
    mpc_result = solve_and_check(mpc_example)
    assert (mpc_result["n_qp"], mpc_result["m_qp"]) == (
        11 * nx + 10 * nu, 22 * nx + 10 * nu,
    )
    print(f"nx={nx}, nu={nu}: n_QP={mpc_result['n_qp']}, "
          f"m_QP={mpc_result['m_qp']}, OSQP={mpc_result['osqp_status']}, "
          f"CVXPY={mpc_result['cvxpy_status']}")
```

这两组分别得到 `(n_QP,m_QP)=(190,300)` 和 `(104,148)`。这些小实例求解成功不代表任意 `nx,nu,seed` 的状态/输入边界都能保证可行。

### 5.7 Eq QP：变量数、约束数、稀疏性、谱和条件数

完整接口为 `EqQPExample(n, m=None, lower_density=0.15, seed=1, max_spectrum=None, max_cond_num=None, r=None)`，生成：

$$
\min_x\;\tfrac12 x^\mathsf{T}Px+q^\mathsf{T}x,
\qquad Ax=b,\qquad P\succeq0.
$$

`qp_problem['n']=n`、`qp_problem['m']=m`，`P.shape=(n,n)`、`A.shape=(m,n)`，等式通过 `l=u=b` 表示。

| 参数 | 默认值 | 当前含义与取值 |
| --- | --- | --- |
| `n` | 必填 | QP 变量数，使用正整数；默认 `m` 时从 `n>=2` 开始 |
| `m` | `n//2` | 等式约束行数，整数且 `1<=m<=n`；行数与约束矩阵的秩是不同概念 |
| `lower_density` | `0.15` | `(0,1]`；同时指定构造 Hessian 所用下部矩阵 `L` 和约束矩阵 `A` 的非零密度 |
| `seed` | `1` | 局部 RNG 的随机种子 |
| `r` | `n` | Hessian `P` 的秩，整数且 `0<=r<=n` |
| `max_spectrum` | `None` | `P` 的最大特征值上界；`None` 表示不额外设上界，指定时为有限正数；`r=0` 时也允许 0 |
| `max_cond_num` | `None` | `P` 最大与最小**正特征值**之比的上界；指定时为有限数且 `>=1`，要求 `r>0` |

Hessian 通过稀疏矩阵构造：`L.shape=(n-r,r)`，将 `[I_r; L]` 随机重排行得到满列秩的 `G`，再用正对角权重 `Lambda` 形成 `P = alpha * G @ Lambda @ G.T`；正标量 `alpha` 用于谱缩放。条件数上界通过限制 `L` 的范数和对角权重范围实现。生成器构造一对满足最优性条件的原始/对偶解，并据此生成 `q,b`，使秩亏的实例仍然可行且目标有界。

使用时需要区分以下含义：

- `lower_density` 控制 `L` 和 `A`，最终 `P` 的非零比例还取决于 `r` 和矩阵乘法产生的填充。默认 `r=n` 时 `L` 没有行，`P` 为正对角矩阵；此时调节 `lower_density` 只影响 `A`。下面使用 `0<r<n` 展示带下部矩阵的构造。
- `max_spectrum` 和 `max_cond_num` 都是**上界**，实际值可能更小。谱参数限制最大特征值，完整特征值序列由生成过程决定。这两个参数均作用于 Hessian `P`。
- 记 `kappa_+(P)=lambda_max(P)/lambda_min_positive(P)`。`r=n` 时这就是通常的谱条件数；`0<r<n` 时 `P` 有零特征值，完整矩阵的通常条件数为无穷大，有限上界描述的是正谱部分。`r=0` 时 `P=0`，不可指定 `max_cond_num`。
- 同时指定 `max_spectrum=S` 和 `max_cond_num=K` 时，所有正特征值位于 `[S/K,S]`，且 `kappa_+(P)<=K`；端点不保证取到。

以下生成 100 个变量、40 行等式约束、秩为 60 的实例，并检查实际矩阵密度和谱。这里的小矩阵可转为稠密数组检查特征值；大规模使用时应保留稀疏存储。

```python
eq_example = EqQPExample(
    n=100, m=40, lower_density=0.05, seed=1,
    r=60, max_spectrum=10.0, max_cond_num=100.0,
)
eq_qp = eq_example.qp_problem
assert (eq_qp["n"], eq_qp["m"]) == (100, 40)
np.testing.assert_array_equal(eq_qp["l"], eq_qp["u"])

for key in ("P", "A"):
    matrix = eq_qp[key]
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(f"Eq QP {key}: shape={matrix.shape}, nnz={matrix.nnz}, density={density:.4f}")

eigenvalues = np.linalg.eigvalsh(eq_qp["P"].toarray())
positive_eigenvalues = eigenvalues[eigenvalues > 1e-9]
condition_positive = positive_eigenvalues[-1] / positive_eigenvalues[0]
assert eigenvalues[0] >= -1e-9
assert positive_eigenvalues.size == eq_example.r
assert eigenvalues[-1] <= 10.0 + 1e-9
assert condition_positive <= 100.0 + 1e-9
print("P 的秩:", positive_eigenvalues.size)
print("P 的最大特征值:", eigenvalues[-1])
print("P 的正谱条件数:", condition_positive)
```

### 5.8 Lasso：变量/样本数、稀疏性、数据尺度和 L1 正则化

完整接口为 `LassoExample(n, seed=1, m=None, density=0.15, data_scale=1.0, lambda_ratio=0.1)`。先生成数据矩阵 `Ad`、真实系数 `x_true` 和带观测噪声的 `bd`，对应原始问题：

$$
\min_x\;\|A_d x-b_d\|_2^2+\lambda\|x\|_1.
$$

| 参数 | 默认值 | 当前含义与取值 |
| --- | --- | --- |
| `n` | 必填 | 特征数，即原始回归变量 `x` 的长度，使用正整数 |
| `seed` | `1` | 局部 RNG 的随机种子；保留为第二个位置参数 |
| `m` | `100*n` | 样本数，正整数，可独立于 `n` 设置，也支持 `m<n` |
| `density` | `0.15` | `(0,1]`；数据矩阵 `Ad` 的非零密度，`Ad.shape=(m,n)` |
| `data_scale` | `1.0` | 有限正数，同时缩放 `Ad` 和 `bd`（包括观测噪声），保持信噪比 |
| `lambda_ratio` | `0.1` | 有限非负数，决定相对于零解阈值的 L1 正则化强度 |

**变量数与约束数的对应关系。** 原始 Lasso 是无约束问题；QP 转换引入 `m` 个残差变量 `y` 和 `n` 个辅助变量 `t`，按 `[x,y,t]` 排列，使用 `y=Ad@x-bd`、`-t<=x<=t`。因此最终变量数和约束行数均为 `m+2*n`：其中 `m` 行残差等式、`2*n` 行不等式。构造参数 `n,m` 控制的是特征数和样本数，最终 QP 的两种尺寸由上述公式关联。

**矩阵稀疏性。** `density` 控制 `Ad` 的非零模式，`nnz(Ad)` 约为 `m*n*density`。QP 的 `P=diag(0_n,2I_m,0_n)`，所以 `nnz(P)=m`；约束矩阵还包含残差和辅助变量的单位矩阵块，`nnz(A_QP)=nnz(Ad)+m+4*n`。真实系数 `x_true` 的非零模式由内部随机规则生成，解的稀疏程度还受数据和正则化影响。

**L1 正则化强度。** 当前残差平方项没有 `1/2`，因此：

$$
\lambda_{\max}=2\|A_d^\mathsf{T}b_d\|_\infty,\qquad
\lambda=\texttt{lambda\_ratio}\cdot\lambda_{\max}.
$$

这两个值分别保存在 `lambda_max`、`lambda_param`。`lambda_ratio=0` 对应无正则化最小二乘；`lambda_ratio>=1` 时 `x=0` 是最优解。默认 `0.1` 保留原来的相对正则化强度。生成后可用 `update_lambda(lambda_new)` 设置**绝对**正则化系数；例如新的比例为 `0.2` 时调用 `update_lambda(0.2 * example.lambda_max)`，已有 OSQP 对象还需更新 `q`，见第 8 节。

**数据尺度。** 固定其余参数和 seed，将 `data_scale` 乘以 `s>0` 会把 `Ad,bd` 同时乘以 `s`，`lambda_max` 和 `lambda_param` 随之乘以 `s**2`。固定 `lambda_ratio` 时，原始目标函数整体乘以 `s**2`，最优回归系数 `x` 的解集保持一致；QP 系数和残差变量的尺度会变化。若希望固定绝对 lambda，可在缩放后的实例上调用 `update_lambda()`。

以下生成 1000 个特征、2000 个样本、数据矩阵非零密度为 0.005 的实例，并展示尺度和正则化比例的用法。它的最终 QP 有 4000 个变量、4000 行约束；此处只生成和检查数据，求解步骤见第 6 节。

```python
lasso_config = dict(n=1000, m=2000, density=0.005, seed=1,
                    lambda_ratio=0.05)
lasso_base = LassoExample(**lasso_config)
lasso_example = LassoExample(**lasso_config, data_scale=2.0)
lasso_qp = lasso_example.qp_problem
assert (lasso_qp["n"], lasso_qp["m"]) == (4000, 4000)
assert lasso_example.Ad.nnz == 10000
assert lasso_qp["P"].nnz == 2000
assert lasso_qp["A"].nnz == 16000
np.testing.assert_allclose(lasso_example.Ad.data, 2.0 * lasso_base.Ad.data)
np.testing.assert_allclose(lasso_example.bd, 2.0 * lasso_base.bd)
np.testing.assert_allclose(lasso_example.lambda_param,
                           4.0 * lasso_base.lambda_param)

print("Lasso Ad:", lasso_example.Ad.shape, "nnz:", lasso_example.Ad.nnz)
print("最终 QP:", f"n_QP={lasso_qp['n']}, m_QP={lasso_qp['m']}")
print("lambda_max:", lasso_example.lambda_max)
print("lambda (ratio=0.05):", lasso_example.lambda_param)

lasso_example.update_lambda(0.2 * lasso_example.lambda_max)
np.testing.assert_allclose(lasso_qp["q"][-lasso_example.n:],
                           lasso_example.lambda_param)
assert lasso_example.cvxpy_param.value == lasso_example.lambda_param
print("更新为 ratio=0.2 后的 lambda:", lasso_example.lambda_param)
```

### 5.9 Huber：分块结构、秩、奇异值和噪声

完整接口为 `HuberExample(n, m, num_blocks, block_sizes, r, max_singular_val, max_cond_num, delta, sigma, outlier_fraction, outlier_scale, seed=1)`。除 `seed` 外，所有参数均必填。生成的数据矩阵为 `Ad.shape=(m,n)`，目标为：

$$
\min_x\;\sum_{i=1}^{m}\phi_\delta((A_dx-b_d)_i),\qquad
\phi_\delta(t)=
\begin{cases}
\tfrac12t^2,&|t|\leq\delta,\\
\delta|t|-\tfrac12\delta^2,&|t|>\delta.
\end{cases}
$$

| 参数 | 当前含义与取值 |
| --- | --- |
| `n`, `m` | 正整数，分别为特征数和样本数，可独立设置，支持 `m<n` |
| `num_blocks` | 正整数，块数量，必须等于 `len(block_sizes)` |
| `block_sizes` | 每项为 `(m_j,n_j,r_j)`，表示块的样本数、特征数和秩；均为正整数，`1<=r_j<=min(m_j,n_j)`；逐项求和必须为 `(m,n,r)` |
| `r` | `Ad` 的目标秩，等于各块秩之和；它与 QP 中名为 `r` 的辅助向量是不同概念 |
| `max_singular_val` | 有限正数 `S`，控制 `Ad` 的最大奇异值；作用于回归数据矩阵，不是 QP Hessian |
| `max_cond_num` | 有限数 `C>=1`，限制最大与最小**正奇异值**之比；秩亏矩阵的零奇异值不参与这个比值 |
| `delta` | 有限正数，Huber 损失从二次段转为线性段的残差阈值 |
| `sigma` | 有限非负数，普通样本的零均值高斯噪声标准差 |
| `outlier_fraction` | `[0,1]` 内的数，每个样本被选为异常样本的概率；实际异常样本比例会随机变化 |
| `outlier_scale` | 有限非负数，异常噪声为 `outlier_scale * Uniform(0,1)` |
| `seed` | 默认 `1`，局部 RNG 的随机种子，不改变 NumPy 全局随机状态 |

**分块与谱的构造。** 第 `j` 块为 `U_j @ diag(s_j) @ V_j.T`，其中 `U_j.shape=(m_j,r_j)`、`V_j.shape=(n_j,r_j)`，通过 QR 分解生成正交列。各块按 `block_sizes` 顺序组装为 CSC 块对角矩阵 `Ad`。当 `r>1` 时，先生成全局正奇异值序列：

$$
s_i=S\,C^{-i/(r-1)},\qquad i=0,\ldots,r-1,
$$

再按每块的 `r_j` 依次分配。因此在精确算术下，最大正奇异值为 `S`，最小为 `S/C`，比值为 `C`。`r=1` 时仅有奇异值 `S`，比值为 1。很小的奇异值可能受到浮点误差影响；当 `r<min(m,n)` 时，包含零奇异值的通常条件数为无穷大，不能把 `max_cond_num` 理解为该条件数的上界。

**稀疏性与计算规模。** 当前没有 `density` 参数，块外元素为零，块内通常稠密。对于稠密块，`nnz(Ad)=sum(m_j*n_j)`；改变分块形状会改变非零比例和每次 QR/矩阵乘法的规模。生成过程只对单个块进行稠密运算，不对完整 `Ad` 做 SVD；各块的回归变量和样本互不耦合。

`generation_stats` 保存 `nnz_U/nnz_V/nnz_Ad`、`density_U/density_V/density_Ad` 和对应的 `sparsity_*`，其中 `sparsity=1-density`。`U,V` 的统计对应形状 `(m,r)`、`(n,r)` 的隐式块对角因子；生成器没有保存完整的 `U,V` 数组。非零数按实际数值统计，不使用阈值截断。

**噪声和 QP 规模。** 先生成 `x_true ~ N(0,I)/sqrt(n)`，再令 `bd=Ad@x_true+noise`。普通样本采用 `sigma` 指定的高斯噪声；异常样本采用上述非负均匀噪声，替换普通噪声。`x_true` 是数据生成参数，并非含噪回归问题的已知最优解。QP 按 `[x,z,r,s]` 排列，目标为 `0.5*||z||²+delta*sum(r+s)`，约束为 `Ad@x-bd-z=r-s, r>=0, s>=0`，因此 `n_QP=n+3*m`、`m_QP=3*m`；样本数不再固定为 `100*n`。

下面复用第 4 节的两块示例：`n=20,m=100,r=12`，块为 `(60,12,8)` 和 `(40,8,4)`。最终 QP 有 320 个变量、300 行约束，`Ad` 的非零数为 1040。代码检查小矩阵的实际谱、生成统计，并求解核对 Huber 损失；这里的稠密 SVD 仅用于验证小示例。

```python
huber_example = instances["huber"]
huber_qp = huber_example.qp_problem
assert (huber_qp["n"], huber_qp["m"]) == (320, 300)
assert sp.isspmatrix_csc(huber_example.Ad)
assert huber_example.generation_stats["nnz_Ad"] == 1040

singular_values = np.linalg.svd(huber_example.Ad.toarray(), compute_uv=False)
positive_singular_values = singular_values[singular_values > 1e-9]
assert positive_singular_values.size == huber_example.r
np.testing.assert_allclose(positive_singular_values[0],
                           huber_example.max_singular_val)
positive_ratio = positive_singular_values[0] / positive_singular_values[-1]
np.testing.assert_allclose(positive_ratio, huber_example.max_cond_num)
print("Huber Ad:", huber_example.Ad.shape, "rank:", positive_singular_values.size)
print("最大奇异值:", positive_singular_values[0], "正奇异值之比:", positive_ratio)
for name, value in huber_example.generation_stats.items():
    print(f"{name}: {value}")

huber_result = solve_and_check(huber_example)
huber_x = huber_example.cvxpy_variables[0].value
huber_residual = huber_example.Ad @ huber_x - huber_example.bd
delta = huber_example.delta
huber_loss = np.where(np.abs(huber_residual) <= delta,
                      0.5 * huber_residual**2,
                      delta * np.abs(huber_residual) - 0.5 * delta**2)
np.testing.assert_allclose(huber_result["osqp_objective"], huber_loss.sum(),
                           rtol=5e-6, atol=5e-6)
print("Huber:", huber_result["osqp_status"], huber_result["cvxpy_status"],
      "objective:", huber_result["osqp_objective"])
```

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

目标值以当前运行输出为准；Lasso 已切换到局部 RNG，同一 seed 的具体数据与历史复现报告不同。`revert_cvxpy_solution()` 将 CVXPY 的原始和对偶解映射回手工 QP 的变量/约束顺序；先成功求解再调用。两条路径都使用 OSQP，主要验证模型转换及解映射，不构成不同求解器之间的独立比较。完整的残差检查在 [tests/test_generators.py](tests/test_generators.py) 中。

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

from problem_classes.tests.test_generators import qp_fingerprint
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

生成器方法不会自动更新用户已创建的 OSQP 对象。Eq QP 的尺寸、密度、秩和谱参数在构造时指定，修改配置时重新实例化。Lasso 的完整更新示例（将绝对 lambda 加倍）：

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

Lasso 更新后目标值由代码打印，并与 CVXPY 模型交叉检查。Control 的新初始状态需由调用者考虑边界及后续动力学可行性，`update_x0()` 不保证任意输入均可行。

## 9. 随机种子和规模限制

Random QP、Eq QP、Lasso 和 Huber 当前使用 `np.random.default_rng(seed)` 局部 RNG，不改变 NumPy 全局随机状态。Control、Portfolio 和 SVM 仍使用 `np.random.seed(seed)`，会改变全局随机状态，不宜在线程中并发调用这些依赖全局 RNG 的构造函数。同一固定依赖环境、全部构造参数和 seed 可复现同一实例；局部 RNG 与旧全局 RNG 的数值序列不同，也不应假设跨 NumPy/SciPy 版本仍逐位一致。

常规求解验证使用表中小规模配置，参数测试还包含 Lasso 的 `n=10000,m=20000,density=0.0005` 稀疏生成检查（只生成，不求解）。这些检查不代表原始论文的全部 1400 个实例或超大规模性能已经验证。放大尺寸前需考虑：

- Lasso 的样本数默认是特征数的 100 倍，可用 `m` 自定义；最终 QP 的尺寸为 `m+2*n`。Huber 的 `m` 必须显式指定，最终变量数为 `n+3*m`，分块大小决定稠密生成步骤的开销。SVM 仍固定使用 `100*n` 个样本。
- 固定密度时，Lasso 的数据非零数随 `m*n` 增长，应结合 `m,n,density` 控制规模。Eq QP 的 `lower_density` 同时作用于 `A` 和构造 `P` 的下部矩阵；`0<r<n` 时形成 `G @ Lambda @ G.T` 可能产生填充，默认 `r=n` 则得到对角 `P`。Random QP 通过 `P_block_sizes` 控制块内稠密运算的规模，P 的存储量最多为各块大小平方之和；A 的非零数随 `m*n*A_density` 增长。
- Control 当前 `A/B` 为稠密数据的稀疏存储，包含稠密特征分解和 Riccati 方程求解；构造函数可独立指定 `nx,nu`，但 `T=10` 固定。增加 `nu` 也会增大 `B,R` 和 Riccati 求解的开销；稀疏 QP 装配不代表前期生成没有稠密瓶颈。
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
from problem_classes.tests.test_generators import run_suite
quick_result = run_suite(quick=True)
assert quick_result["all_passed"]
assert quick_result["total"] == 11
print(f"{quick_result['passed']}/{quick_result['total']} passed")
```
