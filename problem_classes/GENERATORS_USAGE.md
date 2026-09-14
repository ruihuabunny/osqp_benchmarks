# optimisation instance generators 使用说明

本目录中的 7 类随机 optimisation instance generators 已完成实例生成、OSQP 求解和 CVXPY 模型一致性验证。依赖统一安装在**仓库根目录的 `.venv`**。实测环境和结果见 [复现报告](GENERATORS_REPRODUCTION.md)。

本文件对应 [GENERATORS_USAGE.ipynb](GENERATORS_USAGE.ipynb)。Notebook 保留本文说明，将 `python` 代码块转换为可执行单元；按顺序运行即可。`bash` 代码块是终端操作说明，不会在 Notebook 中自动安装依赖或启动服务。

## 1. 范围与入口

这些 `.py` 文件定义生成器类，**实例在调用构造函数时生成**。直接执行 `python control.py` 等文件只会加载类定义，不会生成、求解或保存实例。各构造函数同时创建 `qp_problem` 和 `cvxpy_problem`，本身不调用求解器。

| 文件 | 调用方式 | 首个参数含义 | 本次验证尺寸 |
| --- | --- | --- | --- |
| [random_qp.py](random_qp.py) | `RandomQPExample(n, seed=1)` | QP 变量数；约束数为 `10*n` | 10、30 |
| [eq_qp.py](eq_qp.py) | `EqQPExample(n, m=None, lower_density=0.15, seed=1, ...)` | QP 变量数；等式数 `m` 默认 `floor(n/2)` | 10、30 |
| [portfolio.py](portfolio.py) | `PortfolioExample(k, seed=1, n=None)` | 因子数 `k`；资产数 `n` 默认 `100*k` | k=3、5；另测 n=60 |
| [lasso.py](lasso.py) | `LassoExample(n, seed=1, m=None, ...)` | 特征数；样本数 `m` 默认 `100*n` | 5、10 |
| [huber.py](huber.py) | `HuberExample(n, seed=1)` | 特征数；样本数 `100*n` | 5、10 |
| [svm.py](svm.py) | `SVMExample(n, seed=1)` | 特征数；样本数 `100*n` | 5、10 |
| [control.py](control.py) | `ControlExample(n, seed=1)` | 状态数 `nx=n`；输入数 `nu=floor(n/2)` | 4、10 |

`seed` 默认均为 1。传入明确的正整数尺寸；Control 和 Eq QP 示例建议从 `n>=2` 开始。Portfolio 若指定资产数，应使用关键字 `n=...`，第二个位置参数是 `seed`。

Eq QP 的第二个位置参数现在是 `m`，请通过关键字指定 `seed`；Lasso 的第二个位置参数仍是 `seed`。Eq QP 的秩和谱参数，以及 Lasso 的 `m`、`density`、`data_scale`、`lambda_ratio` 参数定义和示例见 [README 的构造参数说明](README_zh-CN.md#22-构造参数与实例规模)。

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
| Eq QP | `x` | N | m（默认 floor(N/2)） | 可控秩的半正定 Hessian；`A x=b` |
| Portfolio | `[x,y]` | N+k | N+k+1 | `x.T D x+y.T y-mu.T x/gamma`；`sum(x)=1, F.T x=y, 0<=x<=1` |
| Lasso | `[x,y,t]` | M+2N（默认 102N） | M+2N（默认 102N） | `||y||²+lambda*sum(t)`；`y=Ad x-bd, -t<=x<=t` |
| Huber | `[x,z,r,s]` | N+3M=301N | 3M=300N | `0.5||z||²+sum(r+s)`；`Ad x-bd-z=r-s, r,s>=0` |
| SVM | `[x,t]` | N+M=101N | 2M=200N | `0.5||x||²+0.5*gamma*sum(t)`；`t>=diag(b_svm) A_svm x+1, t>=0` |
| Control | `[vec_F(x),vec_F(u)]` | (T+1)nx+Tnu | 2(T+1)nx+Tnu | 状态/输入/终端二次代价、动力学、初始状态及上下界；T=10 |

Lasso 的残差平方项没有 `1/2`。SVM 的符号和 `1/2` 系数按当前源码列出。Huber 对应阈值 1 的损失：小残差为 `0.5*r²`，大残差为 `abs(r)-0.5`。Control 的状态和输入按时间逐列展开，使用 `order='F'`；当前终端代价保留源码中 Riccati 解与其转置的乘积。

### 5.1 Control：`__init__` 参数与 MPC 模型

`ControlExample.__init__()` 生成控制问题的参数，`_generate_qp_problem()` 读取这些 `self` 属性，将整个预测区间的目标和约束组装成 QP。对正整数输入 `n`，当前实现使用：

$$
n_x=\texttt{self.nx}=n,\qquad
n_u=\texttt{self.nu}=\lfloor n/2\rfloor,\qquad
T=\texttt{self.T}=10.
$$

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

源码会调整 `self.A` 的特征值，使其模小于 1，以控制离散动力学的稳定性。`self.A` 旁的 `the matrix A is for QP constraints` 注释不准确，应按上述状态转移矩阵理解。

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

例如构造参数为 `n=100` 时，当前实现规定 $n_x=100$、$n_u=50$、$T=10$，所以：

$$
s=11\times100=1100,\qquad c=10\times50=500,
$$

$$
\boxed{n_{\mathrm{QP}}=1600,\qquad m_{\mathrm{QP}}=2700}.
$$

| 对象 | `n=100` 时的 shape |
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

这个表按维度公式计算，表示构造后的逻辑尺寸，不表示该规模已经通过求解或性能验证。构造函数输入的 `n=100` 是单步状态维数，输出 `qp_problem['n']=1600` 是整条预测轨迹的决策变量总数。

下面复用第 4 节已生成的 `ControlExample(10, seed=1)`，打印实际 shape；此时应得到 `n_QP=160`、`m_QP=270`。代码使用独立的 `control_example`、`control_qp` 名称，便于后面的 Lasso 示例继续执行。

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

该例的目标值约为 `1025.9754409815434`。`revert_cvxpy_solution()` 将 CVXPY 的原始和对偶解映射回手工 QP 的变量/约束顺序；先成功求解再调用。两条路径都使用 OSQP，主要验证模型转换及解映射，不构成不同求解器之间的独立比较。完整的残差检查在 [tests/test_generators.py](tests/test_generators.py) 中。

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
from problem_classes.tests.test_generators import run_suite
quick_result = run_suite(quick=True)
assert quick_result["all_passed"]
assert quick_result["total"] == 11
print(f"{quick_result['passed']}/{quick_result['total']} passed")
```
