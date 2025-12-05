## 项目总览
- 本项目在 MFEM 上实现了 PBTE 的灰体稳态 S_N + DG 离散，支持双弛豫（Callaway）碰撞，源迭代更新宏观量，输出 VTK 可视化结果。
- 主要文件
  - `src/main.cpp`: 主程序、命令行解析、源迭代驱动、装配与求解、后处理输出。
  - `include/cli.hpp`: 运行参数（维度、S_N 阶数、Kn、网格大小、松弛、输出等）。
  - `include/params.hpp`: 灰体物理与无量纲参数。
  - `include/quadrature_sn.hpp`: 离散角度集（目前内置极简 S2/S4 的占位，易替换为标准 S_N 集）。
  - `include/equilibrium_gray.hpp`: 灰体等效“平衡量”与“漂移平衡量”的线性化表达。
  - `include/callaway_rta.hpp`: Callaway 双弛豫碰撞算子（N/R 两条通道，接口支持分别给定参考平衡量）。
  - `include/dg/transport.hpp`: DG 体项对流装配；内部面上风通量集成器挂接处。
  - `include/dg/flux_upwind.hpp`: 上风数值通量（内部面 + 边界面出流矩阵项）；入流边界 RHS 集成器。
  - `include/gray_sn_steady.hpp`: 每个角度方向的稳态系统构造、线性系统求解（GMRES+GS）。
  - `include/normal_closure.hpp`: 法向动量守恒的漂移闭合（线性化求解 u）。
  - `include/postprocess.hpp`: VTK 输出（温度、角向平均、热通量 qx/qy/qz）。

## 运行与命令行参数（`include/cli.hpp`）
- 核心参数
  - `-d` 维度(2/3)；`-s` S_N 阶数（整数，小阶数会有射线效应）；`-p` DG 多项式阶数。
  - `-vg` 组速度；`-knN/-knR` 正常/阻力散射 Knudsen 数；`-L` 特征长度 L_char。
  - 网格：`-nx/-ny/(-nz)` 与域长 `-Lx/-Ly/(-Lz)`；初始温度 `-T`；漂移 `-ux/-uy/-uz`。
  - 源迭代：`-maxit/-rtol/-relax`（最大步、相对阈值、欠松弛）。
  - 边界：`-bc` 开启等温入流；`-Thot/-Tcold` 温度；`-hot <attr>` 指定热壁的边界属性 ID（其他边界为冷）。
  - 输出：`-vtk/-no-vtk` 开/关 VTK；`-o <prefix>` 输出前缀。
- 示例
  - 非边界测试（内部面上风通量 + 源迭代）：  
    `./build/DG4PBTE -d 2 -s 4 -nx 64 -ny 64 -p 1 -vg 1 -knN 0.2 -knR 0.1 -T 1 -maxit 20 -rtol 1e-6 -relax 0.3 -vtk -o no_bc`
  - 按属性的一热三冷（网格需有多边界属性！）：  
    `./build/DG4PBTE -d 2 -s 8 -nx 64 -ny 64 -p 1 -vg 1 -knN 0.2 -knR 0.1 -T 1 -maxit 50 -rtol 1e-6 -relax 0.3 -bc -Thot 1.2 -Tcold 0.8 -hot 1 -vtk -o iso_bc_attr`

## 变量/类与功能
- `GrayCallawayParams` (`include/params.hpp`): 物理与无量纲参数（维度、v_g、Kn_N、Kn_R、热容）。
- `SNDirections` (`include/quadrature_sn.hpp`): 离散角度集 `{omega[i], weight[i]}`。
- `GrayEquilibrium` (`include/equilibrium_gray.hpp`): 灰体“参考平衡”与“漂移平衡”的线性化（可后续替换为真实 BE 形式）。
- `EquilibriumFields` (`include/callaway_rta.hpp`): N/R 参考平衡函数对象（可方向相关）。
- `CallawayRTA`：RTA 双弛豫碰撞算子
  - `apply(g, eq, dirs, L_char, rhs)`: `C[g]_i = -(g_i-g_R^eq)/τ_R - (g_i-g_N^eq)/τ_N`
  - `tauNormal/Resistive` 用 Kn 与 v_g, L_char 计算。
- `VelocityCoefficient` (`include/dg/transport.hpp`): 常向量系数 `v = v_g * omega`（体项对流）。
- `UpwindFaceIntegrator` (`include/dg/flux_upwind.hpp`): DG 上风数值通量
  - `AssembleFaceMatrix(el1, el2, Tr, elmat)` 内部面：双边上风耦合矩阵。
  - `AssembleBdrFaceMatrix(el, Tr, elmat)` 边界面：出流矩阵项（入流由 RHS 注入）。
- `InflowBoundaryRHS`：入流边界面 RHS 注入（仅 `v·n < 0` 处有效，`g_in` 来自属性→壁温映射或回调）。
- `DirectionSystem` (`include/gray_sn_steady.hpp`): 各方向的稳态系统
  - `Aform`: 体项对流 + 内部/边界上风通量（矩阵）+ 质量项（σR+σN）。
  - `bform`: 体项源项 `σ_R g_R^eq + σ_N g_N^eq` + 入流边界 RHS。
  - `owned_coeffs/owned_vcoeffs/owned_bmaps`: 保存资源所有权避免悬空引用。
  - `solveDirectionSystem`: 组装、形成线性系统、GMRES+Gauss-Seidel 求解、恢复 FE 解。
- `computeNormalDriftLinearized` (`include/normal_closure.hpp`): 线性化的 normal 漂移闭合 `M u = (1/alpha) b`。
- `postprocess::SaveVTK`：保存 `temperature`、`g_avg`、`qx/qy(/qz)` 到 VTK。

## 算法流程（源迭代）
1. 初始化网格、L2(FE) 空间、初始温度场 `T(x)=Tref`。
2. 循环 k=1..maxit：
   - 对每个离散方向 i：
     - 构造方向系统：体项对流(Convection)、内部面上风、边界面上风（出流）、质量项 σ，右端 `σ_R g_R^eq(T) + σ_N g_N^eq(T,u)`，并按边界属性或坐标注入入流 RHS：`g_in = Tw`。
     - 装配并求解，得到 `g_i(x)` 并加权累加角向平均 `<g>(x) = Σ w_i g_i(x)`。
   - 更新温度场：`T_new = (1-relax) T + relax * (Tref * <g>)`。
   - 残差：`||T_new - T_old|| / ||T_old||`，达到 `rtol` 则停止。
3. 计算热通量（灰体近似）：`q ≈ v_g Σ w_i ω g_i(x)`。
4. 输出 VTK：`temperature, g_avg, qx/qy(/qz)`。

## 边界条件（DG 正确做法）
- 内部面：DG 上风通量（`UpwindFaceIntegrator::AssembleFaceMatrix`）。
- 边界面：
  - 出流：仅矩阵上风项（自然外推，无 RHS）。
  - 入流：边界面 RHS 注入 `g_in`。本项目按属性映射 `attr -> Tw`；将 `g_in = Tw` 直接注入，避免再套 BE(T) 线性化造成靠壁方向“拉低”的反向层。
- 网格要求
  - 若按属性区分“一热三冷”，网格需有多个边界属性（如 1/2/3/4）。若只有一个属性，则整圈同温，建议换网格或采用坐标判定热壁测试。

## 如何修改/扩展
- 提升角度分辨率（抑制射线效应）：
  - 在 `include/quadrature_sn.hpp` 增加标准 S_N 集合，命令行 `-s` 映射加载（推荐 S8 起）。
- 替换平衡量（真实 BE）：
  - 在 `include/equilibrium_gray.hpp` / `callaway_rta.hpp` 替换 `resistiveBE(T)` 与 `normalDriftedBE(T, ω, u)`，保证单位与 `g` 一致。
- 改碰撞算子：
  - 修改 `CallawayRTA::apply` 的组合方式，或引入非线性 RTA；同步调整 `EqCoefficient` 与 `bform`。
- 改 DG 通量/稳定项：
  - 在 `UpwindFaceIntegrator` 调整上风通量，或引入小的流线扩散/人工粘性平滑界面。
- 线性求解器调优：
  - 在 `solveDirectionSystem` 更换预条件（如 ILU）或使用 PA/MF；增大 `KDim/MaxIter`。
- 修改源迭代更新：
  - 在 `main.cpp` 源迭代循环处修改 `T_new` 更新公式（能量一致性/非线性映射等）、残差定义。

## 结果结构与输出
- 结果场：
  - 温度 `temperature`: `mfem::GridFunction`。
  - 角向平均 `g_avg`: `mfem::GridFunction`。
  - 热通量 `q`：`qx, qy(, qz)`（灰体近似，可替换为能量加权）。
- 输出：
  - `ParaViewDataCollection`（.pvd + .pvtu/.vtu），前缀由 `-o` 指定；支持高阶输出。
- 可视化建议：
  - DG 场在单元间不连续，若需看“每元均值”，可在 ParaView 用 `Cell Data` 或关闭高阶显示。
  - 射线效应（S2/S4 的条纹）是 S_N 固有伪影，建议升阶 `-s 8` 或做角向滤波。

## 常见问题与对策
- 边界附近“反向层”、靠热壁侧反而偏冷：
  - 入流应与 `g` 同单位：`g_in = Tw`，不要再套 BE(T) 线性化。
  - 边界面用 DG 上风（出流矩阵 + 入流 RHS），不要使用 H1 的 Boundary* 惩罚式叠加。
- “同心/条纹”：
  - 射线效应。提高角度阶 `-s`，或加角向扩散；细化网格/升阶也有帮助。
- 按属性边界无效/崩溃：
  - 网格需有多边界属性；否则无法区分“一热三冷”。
  - 确保网格有多边界属性（如四边 1/2/3/4），当前实现依赖 mesh.bdr_attributes 构建 attr -> Tw 映射；若只有 1，则整圈同温。
