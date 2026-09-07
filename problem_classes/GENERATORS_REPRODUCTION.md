# optimisation instance generators 复现报告

**结果：7 类随机实例生成器已跑通，42 组实例检查和 4 类参数更新检查全部通过，共 46/46。** 本报告对应 [GENERATORS_REPRODUCTION.ipynb](GENERATORS_REPRODUCTION.ipynb)，使用方法见 [GENERATORS_USAGE.md](GENERATORS_USAGE.md)。

记录日期为 **2026-09-07**；完整验证结果生成于 **12:11:34 UTC / 13:11:34 Europe/London**。下列数值来自实际运行，没有把导入成功等同于实例生成或求解成功。

## 1. 复现范围与代码位置

验证对象为 `random_qp.py`、`eq_qp.py`、`portfolio.py`、`lasso.py`、`huber.py`、`svm.py`、`control.py`。每个对象均执行构造、读取 QP、直接用 OSQP 求解、通过 CVXPY+OSQP 求解及恢复原始/对偶解。

按要求排除 `maros_meszaros_data/`、`qplib_data/`、`suitesparse_matrix_collection/`，未遍历、读取、下载或修改这些文件夹。相应数据加载类没有实例化；本次没有运行上级完整多求解器 benchmark。

`third_parties/osqp_benchmarks` 是 Git 子模块，代码修改和本文档位于该子模块工作区。记录时的提交为：

| 位置 | 起始提交 |
| --- | --- |
| 主仓库 | `dcced5ecd06cf65289408c1727cebe6afcfcaae1` |
| osqp_benchmarks 子模块 | `e34a4a6709d4676483cacadedd040394ed0b8051` |

上述提交标识修改前基线；本次修复后的 7 个生成器及验证脚本的 SHA-256 另存于 [generator_validation_results.json](generator_validation_results.json) 的 `source_sha256`。基线实测结果保存在 [generator_validation_baseline.json](generator_validation_baseline.json)。

## 2. 环境和依赖

| 项目 | 实测值 |
| --- | --- |
| Python | 3.12.3，GCC 13.3.0 |
| 平台 | Linux / WSL2 x86_64，内核 6.6.87.2-microsoft-standard-WSL2，glibc 2.39 |
| 虚拟环境 | 仓库根目录 `.venv`，`include-system-site-packages=false` |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| CVXPY | 1.9.2 |
| OSQP | 1.1.3 |
| nbformat / nbclient | 5.11.1 / 0.11.0 |
| ipykernel / JupyterLab | 7.3.0 / 4.6.3 |

原 `.venv` 仅有 pip，缺少数值计算与 Notebook 依赖。首次安装被沙箱 DNS/联网限制阻断；经联网授权后，使用 `.venv/bin/python -m pip install --no-cache-dir ...` 完成安装。没有向系统 Python 或用户 site-packages 安装包。

[requirements-generators.lock](requirements-generators.lock) 用 `pip freeze --all` 记录了包含 pip 在内的 **103 个分发包**。`pip check` 实测输出 `No broken requirements found.`。Jupyter 内核 `osqp-generators` 使用 `--sys-prefix` 安装到 `.venv/share/jupyter/kernels/`。

复现命令，从仓库根目录执行；已有 `.venv` 时跳过第一行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r third_parties/osqp_benchmarks/problem_classes/requirements-generators.lock
.venv/bin/python -m pip --no-cache-dir check
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/validate_generators.py --output /tmp/osqp-generators-rerun.json
```

该版本快照针对本次 Python/Linux 环境；时间、浮点末位和跨平台安装可用性不作逐位保证。此后运行生成器无需联网或下载数据。

## 3. 基线问题及修复

### 3.1 原始构造与求解结果

修改前，对 7 类分别使用 seed=1、尺寸 10（Portfolio 为 k=5）运行。**7/7 均能生成并完成两条求解路径**。因此本次没有把弃用警告描述为执行失败。

Random QP、Eq QP、Portfolio、Lasso、SVM、Control 的 CVXPY 表达式使用 `*` 表示矩阵/向量乘法，产生兼容性警告；Control 的两次 `cvxpy.vec()` 还产生未指定展开顺序的 FutureWarning。Huber 此轮无警告。

### 3.2 Lasso 参数更新错误

`LassoExample(10, seed=1)` 的初始 lambda 为 `11.328206422990391`。先求解，再调用 `update_lambda(2*lambda)`，得到：

| 情况 | 直接 QP 目标值 | CVXPY 目标值 | 绝对差 |
| --- | --- | --- | --- |
| 基线：更新前 | 1025.9754409815448 | 1025.9754409815434 | 1.36e-12 |
| 基线：lambda 加倍后 | 1034.2135226400621 | 1025.9754409815434 | **8.238081658518695** |
| 修复后：lambda 加倍后 | 1034.2135226400615 | 1034.2135226400615 | 0 |

原因是 `_generate_cvxpy_problem()` 虽然创建了 `lambda_cvxpy`，目标函数却引用数值常量 `self.lambda_param`，更新 Parameter 不会改变目标。修复为 `lambda_cvxpy * cvxpy.sum(t)`，并声明非负参数。`update_lambda()` 先验证新值再改 QP，负值失败后两种模型维持原状态。复现实验还确认更新后的 QP 与根据当前属性重新装配的 QP 完全相同。

### 3.3 兼容性修复与保持的接口

- 6 个生成器中的矩阵乘法改为 `@`，求和使用 `cvxpy.sum()`。
- Control 的解向量展开显式指定 `order='F'`，与手工 QP 的时间排列一致。
- 保留构造函数签名、`qp_problem` 字段、CVXPY 模型、解恢复与更新方法。
- 保留原随机生成流程、Control 的动力学/代价/边界以及稀疏 QP 装配，没有修改 OSQP 求解器。

另外对下面 42 个配置比较修改前后的全部 `qp_problem` 字段，包括拆分的边界数据：归一化稀疏存储后 **42/42 SHA-256 完全一致**。这说明本次修复未改变这些配置的初始优化实例。

## 4. 验证设计与判定标准

[validate_generators.py](validate_generators.py) 显式导入 7 个随机生成器，不扫描数据目录。默认配置如下：

| 类别 | 首个构造参数 | seeds | 基础检查数 |
| --- | --- | --- | --- |
| Random QP | n=10、30 | 0、1、2 | 6 |
| Eq QP | n=10、30 | 0、1、2 | 6 |
| Portfolio | k=3、5，n=100k | 0、1、2 | 6 |
| Lasso | n=5、10 | 0、1、2 | 6 |
| Huber | n=5、10 | 0、1、2 | 6 |
| SVM | n=5、10 | 0、1、2 | 6 |
| Control | nx=4、10，nu=floor(nx/2)，T=10 | 0、1、2 | 6 |

每个基础配置额外重建相同 seed 的实例，并用 `seed+100` 生成一个对照，仅比较数据指纹；对照实例不计入 42 个求解检查。相同 seed 数据一致，改变 seed 后数据不同。

求解设置为 `eps_abs=eps_rel=1e-7`、`max_iter=100000`、`polishing=True`、`adaptive_rho_interval=50`、`verbose=False`；CVXPY 使用 `warm_start=False`。直接 OSQP 必须返回 `solved`，CVXPY 必须返回 `optimal`。

每个配置还检查：

1. 稀疏矩阵尺寸、有限实数系数、向量尺寸、合法无穷界、`l<=u` 和 Hessian 对称性；CVXPY 模型满足 DCP。
2. 若存在拆分边界字段，将 `A_nobounds` 与单位矩阵的 `bounds_idx` 行重新合并，核对完整 QP 一致。
3. 两条求解路径的原始/对偶解形状和有限性，以及直接由原 QP 计算的约束、驻点、互补残差。
4. 手工 QP 的目标、CVXPY 目标及恢复后的解代入原 QP 的目标一致。
5. 小规模 Control 的动力学谱半径小于 1，初始状态满足状态边界。

设 `a=A x`，无穷范数记作 `||.||∞`，独立检查所用的归一化残差为：

$$r_p=\frac{\max(0,\max_i(l_i-a_i),\max_i(a_i-u_i))}{1+\|a\|_\infty},$$

$$r_d=\frac{\|P x+q+A^\mathsf{T}y\|_\infty}{1+\max(\|Px\|_\infty,\|q\|_\infty,\|A^\mathsf{T}y\|_\infty)}.$$

互补检查使用 `min(max(y,0),abs(u-a))` 和 `min(max(-y,0),abs(a-l))` 的最大无穷范数，再除以 `1+max(||a||∞,||y||∞)`；它也检查只有单侧边界时对偶变量的符号。目标差除以 `1+max(abs(f_QP),abs(f_CVXPY))`。上述归一化检查的统一阈值为 **5e-6**，与求解器自身停止准则分开记录。

额外 4 类参数检查均在更新前后求解，并验证更新后的 QP 与重新装配结果一致：

| 检查 | 参数变化 | 更新后目标值 | 结果 |
| --- | --- | --- | --- |
| `lasso_lambda` | n=10，seed=1，lambda 加倍；另测拒绝负 lambda 后状态不变 | 1034.2135226400615 | 通过 |
| `control_x0` | nx=10，seed=1，x0 缩小至 0.5 倍 | 0.2244629558765981 | 通过 |
| `portfolio_mu` | k=3，n=60，seed=1，mu 加 `linspace(0,0.2,n)` | -1.8119784200687399 | 通过 |
| `portfolio_F_D` | k=3，n=60，seed=1，F 乘 0.9、D 乘 1.1 | -1.6812042177811142 | 通过 |

这 4 项各自包含一次更新前检查和一次更新后检查，计为 4 个更新用例。

## 5. 实测结果

下表取 seed=1 的代表实例；P 的非零数统计完整对称矩阵的存储条目。

| 类别 | 首个参数 | n_QP | m_QP | nnz(P) | nnz(A) | 目标值（约） |
| --- | --- | --- | --- | --- | --- | --- |
| Random QP | 10 | 10 | 100 | 36 | 150 | 3.066253684801336 |
| Eq QP | 10 | 10 | 5 | 36 | 8 | -47.72327095204226 |
| Portfolio | 5 | 505 | 506 | 505 | 2255 | -2.256884289504506 |
| Lasso | 10 | 1020 | 1020 | 1000 | 2540 | 1025.9754409815434 |
| Huber | 10 | 3010 | 3000 | 1000 | 6500 | 286.7334729028137 |
| SVM | 10 | 1010 | 2000 | 10 | 3500 | 408.29539941710823 |
| Control | 10 | 160 | 270 | 210 | 1770 | 1.0051307208318527 |

所有 46 项的直接 OSQP 状态均为 `solved`，CVXPY 状态均为 `optimal`，捕获到警告的用例数为 **0**。其中 Control(nx=10, seed=1) 的动力学谱半径约为 `0.9919299626797471`。

| 指标 | 全部用例及两条求解路径的最大值 |
| --- | --- |
| 原始约束违反量（绝对值） | 2.887e-15 |
| 驻点残差（绝对值） | 5.800e-13 |
| 互补残差（绝对值，按上文 min 定义） | 2.887e-15 |
| 归一化原始残差 | 9.021e-16 |
| 归一化驻点残差 | 9.111e-14 |
| 归一化互补残差 | 2.605e-16 |
| 归一化两模型目标差 | 2.073e-14 |

记录的 `run_suite()` 墙钟时间为 **1.334 秒**。它包含基础实例的重复/不同 seed 构造、求解、更新检查和验证，不包含 Python 启动、依赖安装和 Notebook 内核启动；并非正式性能 benchmark。JSON 中还分别记录单次生成时间、直接求解时间、CVXPY 建模求解时间和 OSQP 迭代数。直接求解计时不包含 `setup()`，不能拿两列计时直接比较求解器速度。

## 6. 在 Notebook 中重新执行和核对记录

下面的代码从仓库根目录或子目录定位文件，并确保使用 `.venv`。Notebook 的每个 Python 代码单元按顺序运行；终端安装命令保留为说明文本。

```python
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = next(
    p for p in (Path.cwd(), *Path.cwd().parents)
    if (p / "third_parties/osqp_benchmarks/problem_classes/control.py").is_file()
)
assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "请使用仓库 .venv 内核"
BENCHMARK_ROOT = ROOT / "third_parties/osqp_benchmarks"
BASE = BENCHMARK_ROOT / "problem_classes"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

recorded = json.loads((BASE / "generator_validation_results.json").read_text())
assert recorded["all_passed"] and recorded["passed"] == recorded["total"] == 46
for name, expected_hash in recorded["source_sha256"].items():
    assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected_hash, name
print("解释器:", sys.executable)
print("已保存记录:", recorded["timestamp_utc"], recorded["passed"], "/", recorded["total"])
print("源码指纹与保存记录一致")
```

重新求解并对照全部标准 QP 的指纹和目标值，不要求耗时或元数据时间戳一致。输出仅在内存中，保留随本报告保存的原始 JSON。

```python
from problem_classes.validate_generators import run_suite

rerun = run_suite()
assert rerun["all_passed"] and rerun["total"] == 46
old_by_case = {row["case"]: row for row in recorded["results"]}
for row in rerun["results"]:
    previous = old_by_case[row["case"]]
    assert row["qp_sha256"] == previous["qp_sha256"], row["case"]
    np.testing.assert_allclose(row["osqp_objective"], previous["osqp_objective"],
                               rtol=1e-6, atol=1e-6)
print(f"重新执行: {rerun['passed']}/{rerun['total']} passed, {rerun['elapsed_seconds']:.3f}s")
print("全部 QP 指纹及目标值与保存记录一致")
```

以下代码重现第 3 节的“修改前后初始数据不变”检查。它只通过 Git 读取上述基线提交中的 7 个生成器源码，不检出文件、不修改工作区，也不读取排除的数据目录。此项需要保留子模块的 Git 历史；仅复制文档和代码到无 Git 的目录时，可跳过此单元，上一单元的生成/求解验证仍可独立运行。

```python
import subprocess
import types
import warnings
from problem_classes.validate_generators import CASES, qp_fingerprint

BASELINE_COMMIT = "e34a4a6709d4676483cacadedd040394ed0b8051"
compatible = 0
with warnings.catch_warnings():
    # 这里执行旧版 CVXPY 表达式，只在本项历史数据比较中忽略已记录的弃用警告。
    warnings.simplefilter("ignore")
    for cls, sizes in CASES:
        filename = cls.__module__.split(".")[-1] + ".py"
        source = subprocess.check_output(
            ["git", "-C", str(BENCHMARK_ROOT), "show",
             f"{BASELINE_COMMIT}:problem_classes/{filename}"], text=True
        )
        historical_module = types.ModuleType("historical_" + filename[:-3])
        exec(compile(source, filename, "exec"), historical_module.__dict__)
        original_cls = getattr(historical_module, cls.__name__)
        for size in sizes:
            for seed in (0, 1, 2):
                before = qp_fingerprint(original_cls(size, seed=seed).qp_problem)
                after = qp_fingerprint(cls(size, seed=seed).qp_problem)
                assert before == after, (filename, size, seed)
                compatible += 1
assert compatible == 42
print(f"修复前后初始 QP 数据一致: {compatible}/42")
```

## 7. 结论边界与交付文件

本次已确认这些小规模实例的生成、导出格式、两种模型的求解与解恢复、固定 seed 复现和 4 类参数更新可用。两条求解路径都使用 OSQP，独立残差检查验证其在原 QP 上的数值条件；本报告没有独立比较其他求解器。

原始论文完整实验的 1400 个实例、全部尺寸、商业求解器和三个外部数据目录均未运行。本轮未扩容生成器：固定密度矩阵、Gram 矩阵填充、Control 的稠密特征分解和 Riccati 求解仍是潜在规模瓶颈。当前全局 NumPy RNG 的行为也保留；同 seed 的逐位一致性限定在本次依赖环境。

| 文件 | 用途 |
| --- | --- |
| [GENERATORS_USAGE.md](GENERATORS_USAGE.md) / [GENERATORS_USAGE.ipynb](GENERATORS_USAGE.ipynb) | 安装、生成、求解、保存/加载、参数更新与 Notebook 使用 |
| [GENERATORS_REPRODUCTION.md](GENERATORS_REPRODUCTION.md) / [GENERATORS_REPRODUCTION.ipynb](GENERATORS_REPRODUCTION.ipynb) | 本次复现过程、修复证据、实测结果及可重跑检查 |
| [validate_generators.py](validate_generators.py) | 独立复现入口；失败返回非零退出码 |
| [requirements-generators.lock](requirements-generators.lock) | 本次 `.venv` 的完整依赖快照 |
| [generator_validation_baseline.json](generator_validation_baseline.json) | 修改前 7 类代表实例及 Lasso 更新错误的原始测量 |
| [generator_validation_results.json](generator_validation_results.json) | 修改后 46 项测量、环境、设置及源码/实例指纹 |

两份 Notebook 使用 `osqp-generators` 内核，均由对应 Markdown 按原顺序转换：仅将 `python` fenced code blocks 变成代码单元，其余文字和终端指令保留为 Markdown 单元。交付前使用 `.venv` 内核从空状态执行全部代码单元，并保留实际输出；同时检查 Notebook 格式、代码单元无错误及与 Markdown 的内容对应关系。
