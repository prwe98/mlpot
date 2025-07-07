# MLPot 深度数学与物理原理总结

## 1. 物理动机与对称性

分子势能面（PES）必须满足物理对称性：

#### 1. 平移不变性（Translational Invariance）
**物理意义**：分子的总能量只与原子之间的相对位置有关，而与整个分子在空间中的绝对位置无关。
**数学表达**：
$$
E(\{\mathbf{r}_i + \mathbf{t}\}) = E(\{\mathbf{r}_i\})
$$
**推导说明**：
- 所有原子坐标加上同一个向量 $\mathbf{t}$，原子间距离 $|\mathbf{r}_i - \mathbf{r}_j|$ 不变。
- 势能函数 $E$ 只依赖于这些距离，因此能量不变。

#### 2. 旋转等变性（Rotational Equivariance/Invariance）
**物理意义**：
- 能量是标量，对任意空间旋转都不变（旋转不变性）。
- 力是矢量，旋转分子时，力的方向也应随之旋转（旋转等变性）。
**数学表达**：
$$
E(\{R\mathbf{r}_i + \mathbf{t}\}) = E(\{\mathbf{r}_i\})
$$
$$
\mathbf{F}_i(\{R\mathbf{r}_j + \mathbf{t}\}) = R\mathbf{F}_i(\{\mathbf{r}_j\})
$$
**推导说明**：
- 能量 $E$ 只依赖于原子间距离，旋转不会改变距离，因此能量不变。
- 力的定义为 $\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E$。对于旋转后的坐标 $\mathbf{r}_i' = R\mathbf{r}_i$，有
$$
\frac{\partial}{\partial \mathbf{r}_i'} = R \frac{\partial}{\partial \mathbf{r}_i}
$$
因此
$$
\mathbf{F}_i' = -\nabla_{\mathbf{r}_i'} E = -R \nabla_{\mathbf{r}_i} E = R \mathbf{F}_i
$$
即力矢量随旋转变换。

#### 3. 置换不变性（Permutation Invariance）
**物理意义**：对于同种原子的交换，分子的物理性质（能量、力）不应发生变化。
**数学表达**：
对于任意同种原子的置换 $\pi$，
$$
E(\{Z_{\pi(i)}, \mathbf{r}_{\pi(i)}\}) = E(\{Z_i, \mathbf{r}_i\})
$$
其中 $Z_i$ 为原子类型。
**推导说明**：
- 只要原子类型和相对位置不变，交换标签不会影响能量和力。
- 这要求神经网络的输入处理和聚合操作必须对原子顺序不敏感（如使用sum/mean pooling）。

## 2. 等变图神经网络的数学结构

### 2.1 特征类型

- **标量特征**（如能量、原子嵌入）：旋转下不变
- **矢量特征**（如力、边方向）：旋转下变换

**数学定义**：
- 标量 $s$ 旋转下 $s \to s$
- 矢量 $\mathbf{v}$ 旋转下 $\mathbf{v} \to R\mathbf{v}$
- 更高阶张量 $T$ 旋转下 $T \to R T R^T$（如偶极矩、力常数矩阵等）

### 2.2 消息传递机制

#### 消息计算

对于每条边 $(i, j)$，定义：
- 距离: $r_{ij} = |\mathbf{r}_i - \mathbf{r}_j|$
- 单位方向: $\hat{\mathbf{r}}_{ij} = (\mathbf{r}_i - \mathbf{r}_j)/r_{ij}$
- 径向基展开: $\phi_k(r_{ij})$，如高斯基函数

**消息函数的等变性要求**：
- 标量消息 $m_{ij}^s$ 必须是旋转不变的组合
- 矢量消息 $\mathbf{m}_{ij}^v$ 必须是旋转等变的组合

**具体实现**：
- 标量消息：
  $$
  m_{ij}^s = f_s(h_i, h_j, \phi(r_{ij}))
  $$
  其中 $h_i, h_j$ 为节点标量特征，$\phi(r_{ij})$ 为径向基特征
- 矢量消息：
  $$
  \mathbf{m}_{ij}^v = a_1 \mathbf{v}_i + a_2 \hat{\mathbf{r}}_{ij}
  $$
  其中 $a_1, a_2$ 为可学习的标量权重（可依赖于节点和边特征），$\mathbf{v}_i$ 为节点矢量特征，$\hat{\mathbf{r}}_{ij}$ 为边方向

**推导证明**：
- 旋转 $R$ 作用下，$\mathbf{v}_i \to R\mathbf{v}_i$，$\hat{\mathbf{r}}_{ij} \to R\hat{\mathbf{r}}_{ij}$
- 因此 $\mathbf{m}_{ij}^v \to a_1 R\mathbf{v}_i + a_2 R\hat{\mathbf{r}}_{ij} = R(a_1 \mathbf{v}_i + a_2 \hat{\mathbf{r}}_{ij})$
- 所以 $\mathbf{m}_{ij}^v$ 保证了旋转等变性

#### 消息聚合

- 标量: $h_i' = \sum_j m_{ij}^s$，对旋转不变
- 矢量: $\mathbf{v}_i' = \sum_j \mathbf{m}_{ij}^v$，对旋转等变

**证明**：
- 旋转后 $\mathbf{m}_{ij}^v \to R\mathbf{m}_{ij}^v$，所以 $\mathbf{v}_i' \to R\mathbf{v}_i'$

### 2.3 等变性保证的数学规则

- 标量 × 标量 → 标量（如 $a \cdot b$）
- 标量 × 矢量 → 矢量（如 $a \mathbf{v}$）
- 矢量 × 矢量 → 标量（点积 $\mathbf{v}_1 \cdot \mathbf{v}_2$，旋转不变）或矢量（如线性组合）
- 更高阶：可用张量积、Clebsch-Gordan系数等实现高阶等变

### 2.4 层的实现与推导

#### 消息层（以E2GNN为例）

核心公式：
$$
\mathbf{m}_{ij}^v = w_1 \mathbf{v}_i + w_2 \hat{\mathbf{r}}_{ij}
$$
$$
m_{ij}^s = w_3
$$
其中 $w_1, w_2, w_3$ 均为依赖于节点和边特征的可学习标量。

**旋转推导**：
- $\mathbf{v}_i \to R\mathbf{v}_i$
- $\hat{\mathbf{r}}_{ij} \to R\hat{\mathbf{r}}_{ij}$
- $\mathbf{m}_{ij}^v \to w_1 R\mathbf{v}_i + w_2 R\hat{\mathbf{r}}_{ij} = R(w_1 \mathbf{v}_i + w_2 \hat{\mathbf{r}}_{ij})$
- 所以聚合后 $\mathbf{v}_i' = \sum_j \mathbf{m}_{ij}^v \to R\mathbf{v}_i'$

#### 更新层

- 计算矢量范数 $||\mathbf{v}_i||$，它是旋转不变的，可与标量特征拼接：
  $$
  \text{mixed} = \text{MLP}([h_i, ||\mathbf{v}_i||])
  $$
- 门控机制 $\text{gate} = \tanh(x_3)$ 控制信息流
- 标量更新 $h_i' = x_2 / \sqrt{2} + h_i \cdot \text{gate}$
- 矢量更新 $\mathbf{v}_i' = x_1 \mathbf{v}_1$

**推导**：
- $||\mathbf{v}_i||$ 旋转下不变，拼接后通过MLP输出标量
- 矢量更新部分 $x_1 \mathbf{v}_1$，$\mathbf{v}_1$旋转下变换，$x_1$为标量，整体等变

### 2.5 径向基函数与包络

- 高斯基函数：
  $$
  \phi_k(r) = \exp(-\gamma (r - \mu_k)^2)
  $$
- 多项式包络：
  $$
  f(r) = 1 + a(r/r_c)^p + b(r/r_c)^{p+1} + c(r/r_c)^{p+2}
  $$

**物理意义**：
- 径向基函数用于编码不同距离下的相互作用，包络函数保证在截断半径处平滑衰减，避免物理不连续。

## 3. 全局-局部特征交互

- 分子级特征通过池化（如mean_pool）获得，再与原子特征拼接，实现全局信息流动

## 4. 损失函数与训练目标

- 总损失:  $L = L_E + \lambda L_F$ 
- 能量损失: $L_E = \frac{1}{N} \sum (E_{pred} - E_{true})^2$
- 力损失: $ L_F = \frac{1}{3N} \sum (\mathbf{F}_{pred} - \mathbf{F}_{true})^2 $
- $\lambda$ 通常取较大值（如100），以保证力的精度

## 5. 物理解释与可解释性

- 径向基函数：物理上对应不同距离下的相互作用分量
- 矢量特征：可解释为原子间的方向性作用（如力、偶极矩）
- 全局特征：捕捉分子整体的物理量（如总能量、总动量）

## 6. 等变性数值检验

- 旋转输入分子结构，能量应不变，力应随旋转变换：

```python
energy_invariant = torch.allclose(E, E_rot, atol=1e-5)
force_equivariant = torch.allclose(R @ F, F_rot, atol=1e-5)
```

## 7. 与主流等变GNN的对比

- SchNet: 只处理标量，不能直接预测力方向
- DimeNet/NequIP: 引入角度/高阶张量，表达力更强但更复杂
- E2GNN/MLPot: 直接处理矢量，结构清晰，物理解释性强

## 8. 总结

MLPot通过严格的等变性设计，实现了物理一致、可解释且高效的分子势能与力预测。其核心在于：

- 消息传递和特征更新均严格区分标量/矢量
- 所有操作均保证旋转等变性
- 训练目标兼顾能量和力，适合分子动力学等高精度场景