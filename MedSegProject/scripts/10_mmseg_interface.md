# 组内 MMSeg 代码需要补的接口

拿到代码后，先不要改模型，先把下面四件事做到。全部是"加开关、加导出"，不改变默认行为。

## A. 数据入口统一用 manifest

训练 / 验证 / 测试全部从 `02_make_manifests.py` 生成的 CSV 读取（`image, mask, dataset, domain_id, split`）。
`domain_id` 就是 ADMD 训练时用的域标签；`-1` 表示未知（只在测试集出现）。
分支数 K 自动取训练集中 `domain_id` 的不同取值个数，不要写死 2。

## B. 命令行开关（训练与推理共用）

| 开关 | 取值 | 作用 |
|---|---|---|
| `--decoder` | `admd`（默认）/ `sam` | `sam` = 用 SAM 原始掩码解码器替换 ADMD，其余（MLME、FDM、损失、两阶段训练、学习率）完全不变。这是 **E1 的受控对照** |
| `--admd_query` | `on`（默认）/ `off` | 关闭域感知查询调制（式 2），Q_domain = Q_prompt |
| `--admd_align` | `on`（默认）/ `off` | 关闭动态特征对齐（式 3），h = F_img |
| `--admd_fusion` | `soft`（默认）/ `hard` / `oracle` / `uniform` / `random` | 见 C |
| `--domain_loss_w` | float，默认按现有代码 | 域分类交叉熵权重；`0` = 不监督域分类器（用于说明域标签监督是否必要） |
| `--dump_gating` | 路径 | 推理时导出 gating.csv（见 D） |
| `--prompt` | `1pt / bbox / 1pt+bbox / randpt / bbox_jit5 / bbox_jit10 / bbox_jit20 / noprompt` | 提示从 `03_prompts.py` 的 JSON 读取，**不要在模型代码里自己算质心**，保证所有方法用同一组提示 |
| `--seed` | int | 同时设置 python / numpy / torch / cudnn 的种子，并写进输出目录名 |

## C. 融合策略的实现位置

大论文式 (3.11)：M_final = Σ_k softmax(W_adapt · h)_k · M_k。在算出 K 维权重 w 之后加一个分支：

```python
w = softmax(W_adapt @ pool(h))            # [B, K]，现有代码
if fusion == 'hard':    w = one_hot(w.argmax(-1), K)
elif fusion == 'oracle': w = one_hot(domain_id, K)        # 仅源域测试集有意义；目标域跳过
elif fusion == 'uniform': w = full_like(w, 1.0 / K)
elif fusion == 'random':  w = one_hot(randint(K), K)      # 用 --seed 固定
```

注意：`pool(h)` 现有代码怎么把特征图变成向量（平均池化？取某个 token？）要在论文方法节写清楚，这一处也是 `[CONFIRM]` 之一。

## D. gating.csv 导出格式

推理时每张图一行：

```
image, domain_true, d_0, ..., d_{K-1}, alpha, w_0, ..., w_{K-1}
```

- `d_k`：式 (3.8) 域分类器 softmax 输出
- `alpha`：式 (3.10) 的 α_domain 实际取值（如果是向量就取均值，并在论文里说明）
- `w_k`：式 (3.11) 最终融合权重
- `domain_true`：来自 manifest 的 `domain_id`

`08_gating_analysis.py` 直接消费这个文件。

## E. 预测输出格式

`{out}/masks/{image_stem}.png`，0/255，**恢复到原图分辨率**（最近邻），与 `04_metrics.py` 对接。
不要只保存 1024×1024 的结果，否则 HD95 的像素单位与其他方法不可比。

## F. 必须向项目组确认并写进论文的 8 个问题

1. 域分类器（式 3.8）前有没有梯度反转层？初稿图 1 画了，正文没有。
2. 域分类器用什么损失训练、权重多少？训练时是硬路由（只激活真值域分支）还是软加权？
3. α_domain 的定义（是 max_k d_k？可学习标量？随置信度的函数？）。
4. F_mamba ∈ R^{N×D} 如何变成 R^D 再过线性层。
5. d ∈ R^K 如何与 Q_prompt ∈ R^{N_token×256} 做"逐元素乘"（有投影层？广播到哪一维？）。
6. 第二阶段微调到底用的什么数据（ClinicDB/ISIC 训练划分，还是 SAM-Med 0.1% 子集，还是两者），各多少张；SAM-Med 子集的文件清单。
7. 大论文表 4.3/4.19 的 MMSeg（93.51）和表 4.7 的 MMSeg（92.67）是不是同一个权重、同一提示模式。
8. 原始推理日志（逐图 Dice/HD）还在不在；HD 当时按什么分辨率、什么单位算的；F1 是怎么算的。
