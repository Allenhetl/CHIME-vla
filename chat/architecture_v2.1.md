# 长程 Memory VLA 架构 v2 — CHIME-VLA

> **文档定位**。本文是对 `memory_vla_direction_howtorember.md`(HC-1..HC-8)与 `memory_vla_direction_whentoremember.md`(§3.1..§3.4)两份 problem statement 的架构应答。**不是**待执行的方案,是模块化、可拆解评估、可独立验证的设计文档。读者应能孤立地阅读任何一个组件卡片。
>
> **命名**。CHIME = **C**ausal **H**indsight **I**mmutable dual-channel **ME**mory。v2 与 v1(CW-Mem)的延续与差别在 §A 中显式声明。
>
> **本文档已通过三轮独立 adversarial review**(技术严苛性 / 遗漏元素 / 工程可行性),对 11 处技术不一致与 8 处遗漏做了显式修正。审查发现仍存在的 trade-off 在 §H 中显式承认。

---

## §0 方案总览:数据如何流通,谁在解决哪个问题(完整讲清)

> 本节是 §A–§I 的端到端 narrative。读完这一节,读者应当能在不读任何组件卡片的前提下,讲清:(a) 一帧观测进入系统后,数据走哪条路,(b) 每个模块对哪条 HC/§ 起作用,(c) 训练时与部署时的数据流差别在哪里。

### §0.1 故事性 walkthrough:抽屉 + 红块任务,一步步看数据流

任务:"打开抽屉,把红块放进去,关上抽屉",T=100 步。

**Step 1 — 观测进入**。每步 t,机器人接到 RGB 图像 + proprioception(o_t)。**[C1] VLM 主干**(SigLIP-ViT-L,frozen + LoRA)把它编码为 256 个 token、每个 1152 维的 hidden state h_t。这一步与所有现有 VLA 一样,**不解决任何 HC/§**,只是把 raw 观测变成可处理的特征。

**Step 2 — 短期缓冲**。h_t 进入 **[C2] FIFO 工作缓冲 M_work**——一个固定 8 帧的环形 buffer,把最老的踢出、最新的加入。这一步对应短程 VLA 的 history,**承担 HC-1 的短期容量**(< 8 步内的细节都在),且因为没有可学门控,梯度沿 cross-attn 走 8 步内全可达(HC-2 在短窗内 trivially 满足)。**注意**:它不豁免 HC-3——读端 cross-attn 仍可被训成"只看最新一帧",所以加 L_aux attention entropy 正则做对策。

**Step 3 — 决定要不要把这一帧"钉下来"**。这是**全文档的核心环节**——对应 §3.1"写入时无法预知未来需要什么"+ HC-5"边界检测器单点故障"。**[C5] ESPC** 用一个 1-layer Transformer ψ 看 M_work 预测 ĥ_t|t-1,然后把 h_t 与 ĥ 比对算预测误差:
- e_geo_t(局部空间投影差):捕捉"几何状态突变",对应物体位置/手部动作的物理变化
- e_sem_t(全局语义投影 cosine 差):捕捉"任务进度突变",对应子任务边界

经 EMA 标准化 + sigmoid 得 **γ_geo_t 与 γ_sem_t ∈ [0,1]**——两个连续标量,告诉下游"这一帧的几何信息有多值得记 / 语义信息有多值得记"。**γ 是连续标量而非 0/1 分类**,这是回应 HC-5 的关键(单一分类器误差会被任何上限封顶,连续 soft signal 不会)。

**关键**:γ 在被下游用之前先经 stop-grad(sg)。这是回应 HC-3 的关键——L_main 不能流回 [C5] 训写入门,否则 truncated BPTT 会把 γ 训成全 1(短视)。

在 t=20(手伸向把手,几何在加速),γ_geo_20 ≈ 0.6,γ_sem_20 ≈ 0.15;在 t=60(红块刚放下,语义状态突变),γ_sem_60 ≈ 0.85。**γ_sem 在子任务边界附近自然升高,γ_geo 在状态突变附近升高——这是按 §3.4 异质精度分轴的具体落地**。

**Step 4 — 写入两个不同精度的 memory**。对应 HC-7(几何亚厘米 vs 语义任务进度,差 3 个数量级)+ §3.4(异质共表示有损)。
- **[C3] Geo 写头 + [C6] M_geo grid**(NICE-SLAM 风格 multi-resolution voxel grid,3 层 8³/16³/32³):token 投影到 3D 坐标,sg(γ_geo) 加权后**delta-rule**(无门控乘子,纯加性外积)写入 grid。承担"亚厘米空间精度"通道。
- **[C4] Sem 写头 + [C7] M_sem slot bank**(64 个 slot × 256 维):pool(h_t) 算 query,与 64 个 frozen key 做 softmax-over-slots 竞争,sg(γ_sem) 加权后**delta-rule**累加到选中 slot 的 value 上。承担"任务进度语义"通道。

**为什么是 delta-rule(M_t = M_{t-1} + Δ)而非门控更新(M_t = g·M̂ + (1-g)·M_{t-1})**:门控更新在常规 g ≤ 0.5 设置下,旧内容乘子 (1-g) 跨 50 步累乘后是 10^{-15} 量级——这是 §3.2 给出的代数趋势(详细论证见 §F 选择 1,强度 [中],承认存在跨域迁移风险)。delta-rule 把旧内容加性保留,跨 200 步内容仍可读。**这是回应 §3.2 与 HC-3 的根本机制**。

容量满后(K_s=64 全部 ‖v_i‖ 超阈值):**LRU 丢弃**最不重要的整个 slot(由 [C12] CSM 的 slot 重要度排序决定),**不做合并平均**——合并平均等价于 g=0.5 的乘性 mixing,会复活 §3.2 的衰减(这是审查 Bug 1 的修正)。代价:被丢弃的 slot 内容彻底丢失(§H Trade-off 5)。

在 t=60,Sem 写头强写入(γ_sem=0.85),slot #18 被分配给"红块入抽屉"事件;Geo 写头同时写入红块当前位置坐标到 fine 层 voxel。**这两条信息此后都不会被衰减、覆盖**(除非容量满 + LRU 丢弃,但 episode 长 100 < K_s=64,不触发)。

**Step 5 — 决策时把记忆读出来**。**[C8] 读出接口**用 N_q=16 个 learnable query 对 (M_work + M_sem) 做 cross-attn,N_geo_q=16 个 spatial query 对 M_geo 三线性采样(coarse 层为主,mid/fine 局部精化以满足 HC-6 实时性)。三路 KV 拼接成 context tokens c_t,送入 **[C9] Action Expert**(π0 风格 flow matching head + LoRA)产出动作 a_t。

在 t=90(关抽屉,红块完全不可见),q_90 与 slot #18 的 key 高余弦相似 → cross-attn 给 slot #18 高权重 → c_90 含"红块在抽屉里"的语义 → action expert 知道"继续关到底,不要重开看"。**这是 belief state 跨 30 步保留的具体实现**,纯靠 delta-rule + immutable bank,不依赖任何长程 BPTT。

**Step 6 — 训练时多了什么(部署时不跑)**。前 5 步是部署 forward 路径。训练时还有 4 个组件给写入端反传压力(对应 HC-4 aux loss 必要性):
- **[C10] HCS-H**(Hindsight Causal Saliency Head):在离线 demonstration 上,用 frozen base policy 算 ∂a*_{t+Δ}/∂o_t(Δ ∈ {4, 16},显存允许时加 64)。"未来 16 步动作对当前帧的敏感度高"= 当前帧值得记 → 推 γ̂_t 高 → L_HCS BCE 把 γ_t 推到 γ̂_t。**这是回应 §3.1(承认无法预知未来,但用离线已知未来反推)+ §3.3(每帧 label,绕过跨百步 BPTT)+ HC-2(给旧帧直接梯度路径)的核心机制**。在 t=60,L_HCS 立刻把 γ_sem_60 推到 0.85+,告诉 ESPC "未来这帧很关键"。
- **[C11] PRH**(Predictive Read Head,MERLIN 风格):从 m_t 预测未来 (o_{t+k}, a*_{t+k}),k ∈ {4, 16, 64}。L_PRH 强制 memory 内容"对未来可预测",这是 §3.1 的另一条代理(承认无法知道"未来需要",但要求"未来可由 memory 预测")。
- **[C12] CSM**(Counterfactual Slot Mask):leave-one-slot-out 算每个 slot 的 KL 重要度 w_i,L_CSM 鼓励 slot 间重要度方差大(异质化)。回应 §3.4。在 t=90,KL(π(a|m_90) ‖ π(a|m_90 \ slot #18)) 应当大——因为忘了 #18 就会重开抽屉。这监督 [C4] 把"红块入抽屉"路由到独立 slot 而非 hash collision。
- **[C13] 反向 Jacobian**(原 reverse replay 修正版):同时算 ∂a*_{t-Δ}/∂o_t 处理反向因果场景(柔性接触),罕见但兜底用。

### §0.2 一张图:数据流走向 + 每条流回应哪条 HC/§

```
                     o_t (RGB + propio)
                            │
                            ▼
                  ┌───── [C1] VLM 主干 ─────┐  ← 不解决 HC/§,只编码
                  │       h_t (256×1152)    │
                  │                          │
       ┌──────────┼─────────┬────────────────┘
       ▼          ▼         ▼
   [C2] FIFO   [C5] ESPC   [C8] Read I/F
   (HC-1 短期) (HC-5 连续 γ│   ▲ ▲ ▲
               + §3.1 代理)│   │ │ │  ← cross-attn 全可微 (HC-2)
                  │  γ_geo │   │ │ │
                  │  γ_sem │   │ │ │
                  ▼   sg   │   │ │ │
              ┌──────────┐ │   │ │ │
              │[C3] Geo │─┼──►│ │ │  ← delta-rule 写入 (§3.2)
              │  写头   │ │   │ │ │
              │   ↓     │ │   │ │ │
              │[C6] grid│─┼───┘ │ │  ← 几何通道 (HC-7)
              └──────────┘ │     │ │
              ┌──────────┐ │     │ │
              │[C4] Sem │─┼─────►│ │  ← softmax-over-slots
              │  写头   │ │     │ │     + delta-rule (§3.2 + HC-3)
              │   ↓     │ │     │ │
              │[C7] bank│─┼─────┘ │  ← 语义通道 (HC-7)
              │  +LRU   │ │       │     LRU 保 delta-rule 代数 (§3.2)
              └──────────┘ │       │
                            ▼      │
                    c_t = read 出来的 context
                            │
                            ▼
                     [C9] Action Expert
                            │
                            ▼
                          a_t

╔═══════ Training-only (部署时不跑) ═══════════════════════╗
║                                                           ║
║  [C10] HCS-H: ∂a*_{t+Δ}/∂o_t → γ̂_t  ──L_HCS──► [C5]      ║
║       回应 §3.1 / §3.3 / HC-2 / HC-4 / HC-5 (部分)        ║
║                                                           ║
║  [C11] PRH: m_t → predict (o_{t+k}, a*_{t+k})            ║
║       ──L_PRH──► [C3][C4] 写头  (HC-4, §3.1 代理)        ║
║                                                           ║
║  [C12] CSM: leave-one-slot-out KL ──L_CSM──► [C4]        ║
║       (HC-4, §3.4 异质化)                                 ║
║                                                           ║
║  [C13] Reverse J: ∂a*_{t-Δ}/∂o_t → γ̂ 兜底 (§3.3 部分)    ║
╚═══════════════════════════════════════════════════════════╝

stop-grad (sg) 关键位置:
• [C3][C4] 写入瞬间 sg(γ_*)  → 阻断 L_main 训写入门 (HC-3)
• [C8] L_PRH 路径 sg(query)  → 阻断 L_PRH 污染 [C1]
• [C10] 输出 sg(γ̂)            → γ̂ 是 target 不是可学量
• [C12] 经 frozen [C9]        → 自然 sg
```

### §0.3 一张表:每个 HC/§ 是哪个模块在起作用,通过什么 loss

| 问题 | 主要模块 | 通过什么机制起作用 |
|---|---|---|
| HC-1 容量充分 | [C2][C6][C7] | 总容量 ~12.9 MB(fp32),远超 100 bits 下界——不是真瓶颈 |
| HC-2 梯度可达 | [C8] 全可微 + [C10] | L1 主任务沿 cross-attn 反传 + L_HCS 直接给旧帧 saliency 梯度,绕开 BPTT |
| HC-3 不可学 mixing | [C7] delta-rule + [C3][C4] sg(γ) | 写入瞬间 sg 阻断 L_main 训写入门,delta-rule 无门控乘子 |
| HC-4 aux loss 必要 | [C5][C10][C11][C12] | 三个独立 aux(HCS / PRH / CSM)给写入端反传保留压力,stop-grad 互相隔离 |
| HC-5 边界检测器单点(部分) | [C5] 连续 γ + [C10] 双 fork 信号 | γ 是连续标量非分类,[C10] 内部 Jacobian + RUDDER 双 fork(同组件内部,不是独立校验,详见 §H Trade-off 6) |
| HC-6 实时性(勉强) | [C2][C6][C7][C8] GPU 并行 | 实测 70-135 ms on H100,A100/4090 可能溢出 |
| HC-7 信息精度异质 | [C3][C4][C6][C7] 双通道 | 几何走 voxel grid(空间索引),语义走 slot bank(content key) |
| HC-8 NLP 模板受限 | 整体 | 只抽原子机制(delta-rule、cross-attn、predictive read),不整体复用 |
| §3.1 因果不可能 | [C5] + [C10] + [C11] | 训练时离线后视监督 + 部署时预测误差代理(部署时退化,§H Trade-off 1) |
| §3.2 几何衰减 | [C7] delta-rule + LRU 丢弃 | 加性更新代数上不衰减(容量未满时);满后丢弃整个 slot 而非合并平均 |
| §3.3 跨步信用 | [C10] 每帧 label + [C13] 双向 J | Jacobian 显存 cap 强制 Δ ≤ 64,极长依赖仍受限(§H Trade-off 7) |
| §3.4 异质共表示 | [C3][C4][C6][C7] 双通道 + [C12] | 结构性双通道 + L_CSM 强制 slot 间因果差异化 |

### §0.4 训练 vs 部署的数据流差别(为什么这个架构是 train-heavy)

**部署时(实时,< 100 ms 预算)**:只跑 [C1] → [C2/C5] → [C3/C4] → [C6/C7] → [C8] → [C9],单向。所有 γ_*_t 来自 [C5] 的 self-supervised prediction error,**没有 L_HCS 校准**(部署时没有未来动作可看)。

**训练时(offline,800-1000 ms/step)**:在部署 forward 之上额外跑:
- [C10] 反向算 Jacobian(需 frozen base policy 跨 Δ 步 forward)
- [C11] 三个 head 预测未来
- [C12] 4 slot leave-one-out × 2 frozen [C9] forward
- L1 + L_HCS + L_PRH + L_CSM(+ L_aux + L_GC 可选)联合优化

**关键约束**:训练时所有 γ_*_t 在写入瞬间被 sg,所以 L_main 不会污染 [C5];L_HCS 在 γ̂ 侧 sg,所以它只训 [C5] ψ 与 projections,不污染主干。**这种 stop-grad 矩阵是本架构正确性的关键**——任何 sg 不一致会让 HC-3 隔离失效,整个架构退化(详见 §B 数据流图的 sg 标注)。

### §0.5 这个方案到底"赌"什么

不是赌"新架构一定 work",而是赌三件事各自的概率:

1. **§3.2 真的可以靠 delta-rule + LRU 丢弃绕过**(选择 1,[中])。如果错,长程任务塌陷,fallback 到类 B 的可学 g(回到 §3.2 困境)。
2. **HCS Jacobian 在真实 expert demo 上信噪比够**(选择 3,[弱])。**E1 的核心赌注**。如果错,disable [C10][C12][C13],架构降级为"prediction error self-supervised + L_PRH"的简化版,仍能打 HC-3 + §3.2 + §3.4 三条,workshop paper 级 publishable。
3. **按精度分通道(grid + slot)优于按位置分(VLM + expert)或单 bank 双 head**(选择 2,[中])。如果错,合并为单 bank,损失 ~10-15% 长程 SR 但架构整体不塌。

赌注按"如果错了,会降级到什么状态"排序:**赌注 2 错 → MVP fallback 路径(§I.2);赌注 1 错 → 整个 §3.2 claim 推翻,需要重新设计;赌注 3 错 → 单通道版本仍 publishable**。E1(M1,week 4)是触发降级路径的判决点。

### §0.6 与现有 Memory VLA 工作的差异(一句话每条)

| 工作 | 它的核心机制 | CHIME 与之差异 |
|---|---|---|
| MemoryVLA | top-k 不可微检索 | CHIME 全可微 read,L_HCS 直接给旧帧梯度 |
| Mem-0 | learned subtask classifier 触发 | CHIME 用连续 prediction error + offline saliency,避免分类器单点 |
| ReMem-VLA | frozen-EMA + POP 整图重建 | CHIME 用 delta-rule(无 EMA mixing)+ MERLIN 风格 future-prediction(非整图重建),且加 L_HCS 给写入端额外信号 |
| Goal2Skill | VLM verifier 慢循环 + 符号记忆 | CHIME 全在 VLA 快循环内(< 100 ms),不依赖 VLM 推理 |
| MEM (π0.6) | 视频短期 + 语言长期 | CHIME 不用语言摘要,几何 / 语义都是 latent;短期=FIFO,长期=delta-rule slot |
| MemER | VLM 主动提名关键帧 | CHIME 写入端纯神经,不依赖 VLM 慢推理 |
| PSM-CWM(原型) | HMD loss(Jacobian saliency) | CHIME 把 HMD 扩展为 Jacobian + RUDDER 双 fork 信号(同组件内部 fork,见 §H Trade-off 6),且配合双通道写入 + delta-rule 不可变 bank |

读完本节,后续 §A–§J 是各个细节的展开。如果只关心"为什么这样设计",§0 已经讲完。如果要落地实施,§I 给出工作流。如果要审查正确性,§B(数据流图 + sg 标注)+ §E(三向映射表)+ §J(审查痕迹)是入口。

---

## §0.7 在你的实际硬件上的可实施性(6×A800 训练 / 4090 48G 推理)

> **首先纠正一处不诚实表述**。文档先前 [C5] 卡片与 §H Trade-off 3 中的"已 benchmark"措辞**是估算,不是实测**。所有以 H100 为参考的 latency 数字(30-50 ms / 5-15 ms / 等)都应理解为基于 FLOPs + 显存带宽的纸面推算,需要在目标硬件上实测确认。本节用一致的口径,把所有数字重新映射到 6×A800 + 4090 48G。

### §0.7.1 训练侧:6×A800-80GB 的可行性

**单卡吞吐对比**(BF16,实际可达):
- H100:~500 TFLOPS BF16(原文档假设)
- A800-80GB:~280-310 TFLOPS BF16(中国合规版,NVLink 带宽限制为 400 GB/s,与 A100-80GB BF16 算力相同,只是 NVLink 慢)
- **A800 单卡 ≈ H100 单卡的 0.55-0.62 倍**

**显存** A800-80GB ≈ H100-80GB,**显存上限相同**——所以 §H Trade-off 3 里 [C10] Δ=64 + frozen base policy 跨步 forward 在 H100 上"OOM 风险 ~30%"的判断,在 A800 上**风险相同**。需要 gradient checkpointing 把 frozen base policy 的 activation 折半。

**6 卡聚合算力**:
- 原计划:4×H100 ≈ 2000 TFLOPS(80% DDP 效率)
- 现实:6×A800 ≈ 1700-1860 TFLOPS(70-75% DDP 效率,NVLink 带宽吃亏)
- **聚合算力 ≈ 原计划的 0.85-0.93 倍**——基本相当,够用

**Epoch 时间重估**(10M frames、batch 32):
- 单 step 计算量(L1 + L_HCS Δ∈{4,16} + L_PRH + L_CSM) ≈ 1045 GFLOPs
- A800 utilization 35% → ~100 TFLOPS effective per card
- 单卡每 step 计算 ≈ 1045/100000 = 10.4 ms;算上 IO + DDP 同步 + frozen base policy 重复 forward,**实际 1.4-1.8 s/step on A800**(H100 上 0.8-1.0 s)
- 312k step / epoch ÷ 6 卡 = 52k step / 卡 / epoch × 1.5 s = **~22 小时/epoch**(H100 4 卡时是 20 小时,因为 6 卡多但每卡慢)
- **1 周训练 = 7 epoch**——比 H100 估的 5-6 epoch 还略好,因为多了 2 张卡

**结论**:6×A800 在训练侧**可行**,且总训练预算与原计划接近。**前提**:
- Δ=64 必须先在小 batch 上验证 OOM,默认配置 Δ ∈ {4, 16}
- gradient checkpointing 必须开
- batch size 上限可能从 32 降到 24(Jacobian 显存占用)

### §0.7.2 推理侧:4090 48G —— **HC-6 在原方案下大概率超时**

这是真正的可行性瓶颈。RTX 4090(48GB 是 modded 版本,中国常见)BF16 算力 ~83 dense / ~100 sustainable TFLOPS,**约 H100 的 1/5**;memory bandwidth ~1 TB/s,**约 H100 的 1/3**——Transformer 推理在小 batch 下主要受 memory bandwidth bound,所以**单步延迟约为 H100 的 3-4 倍**。

**原方案在 4090 48G 上的纸面估算**(再强调:非实测):

| 组件 | H100 估计 | 4090 估计(×3-4) | 说明 |
|---|---|---|---|
| [C1] SigLIP-ViT-L (~600M frozen + LoRA) | 30-50 ms | **100-200 ms** | ViT-L 是延迟主因 |
| [C2] FIFO ring buffer | < 1 ms | < 1 ms | 纯内存操作 |
| [C5] ψ over 2k token (1-layer Transformer) | 5-15 ms | **15-50 ms** | self-attn 2048² × 1152 |
| [C8] cross-attn over (M_work + M_sem + grid 三线性) | 5-10 ms | **15-30 ms** | KV 数 ~2k + 16 query × 8 邻居 voxel |
| [C9] Flow matching ODE(默认 4-8 步) | 30-60 ms | **100-240 ms** | ODE 步数线性放大 |
| **合计单步** | **70-135 ms** | **230-520 ms** | |

**结论**:**原方案在 4090 上控制频率仅 2-4 Hz,远低于 HC-6 要求的 5-10 Hz**。HC-6 在 4090 上**会被破坏**,这是个新的硬性障碍,需要在 §H 升级为 Trade-off 8 的明确要求。

### §0.7.3 推理优化路径(把 4090 48G 做到 HC-6 边界内)

要把 4090 单步压到 100-150 ms 范围(对应 6.7-10 Hz),需要叠加以下优化:

| 优化 | 节省延迟 | 代价 / 风险 |
|---|---|---|
| **[C1] 换为 SigLIP-ViT-B**(~150M,而非 ViT-L) | -60-130 ms | 表征质量降级,可能导致基础 BC 性能 -3-5% SR;但记忆机制效果大概率不变(LoRA 仍补足任务相关特征) |
| **[C9] ODE 步数从 4-8 减到 1-2 步**(consistency model 蒸馏 / single-step flow) | -70-180 ms | 动作精度可能略降;π0 系列已验证 1-step inference 可行 |
| **量化 [C1][C9] 到 INT8 或 FP8** | -30-50 ms | 需要校准数据集,精度损失 <1% |
| **[C5] ψ 简化为 1-layer GRU 而非 Transformer** | -10-30 ms | 预测误差信号可能轻微变粗;e_geo / e_sem 可能损失精度 |
| **Memory KV cache between steps**(M_sem / M_geo 仅在 γ>阈值 时更新,可缓存 [C8] KV 表征) | -5-10 ms | 需要 invalidation 逻辑;实现复杂度 +1 人周 |

**优化后 4090 估算**(全部叠加):

| 组件 | 优化后估计 |
|---|---|
| [C1] ViT-B + INT8 | 30-60 ms |
| [C5] GRU-ψ | 5-15 ms |
| [C8] cross-attn + KV cache | 8-15 ms |
| [C9] 1-step flow + INT8 | 25-50 ms |
| **合计单步** | **70-140 ms** → 7-14 Hz ✅ |

**这条优化路径让 HC-6 在 4090 上重新可达**。但代价是:
1. **MVP 阶段必须从 ViT-B 起步**,而非 ViT-L——所有"基础 SR"对照实验跟着重新跑
2. **基础 BC 性能上限被 backbone 选择封顶**(ViT-B vs ViT-L 在精确抓取任务上的差异在文献中通常 -3-5% SR)
3. **ODE 1-step 蒸馏需要单独的训练阶段**(从 8-step teacher 蒸馏 1-step student),增加 ~1 周训练时间

### §0.7.4 修订后的 MVP 配置(替换原 §I.2)

**硬件锁定**:6×A800 训练 / 4090 48G 推理。

**默认 backbone**:
- [C1] = SigLIP-ViT-B + LoRA r=16(而非 ViT-L)
- [C9] = π0 flow matching head + 1-step consistency 蒸馏
- 推理阶段全部 INT8

**砍掉**(同原 §I.2):[C10][C12][C13]、L_HCS、L_CSM、L_GC;[C6] 单分辨率 16³;benchmark 收敛到 LIBERO-Long + CALVIN ABCD→D。

**保留**:[C1 ViT-B][C2][C3 简化][C4][C5 仅 prediction-error self-supervised,GRU 实现][C6 16³][C7 LRU 丢弃靠简单 timestamp][C8 + KV cache][C9 1-step] + L_main + L_PRH + L_aux。

**预期推理延迟**:80-130 ms(7.7-12.5 Hz) ✅ HC-6 满足

**预期训练时间**:10M frames、batch 24、6×A800、5 epoch ≈ 4-5 天

**MVP 可发表 paper claim**(更新):"基于 event-segmentation prediction error 触发的双通道 delta-rule 不可变 memory + MERLIN 风格 predictive read,**推理 7-12 Hz on consumer-grade GPU**(4090 48G),在 CALVIN ABCD→D + LIBERO-Long 上 SR 比 OpenVLA + history baseline 高 X%。" —— 加入"消费级硬件可部署"作为 contribution 之一。

### §0.7.5 6-month full 版本在你硬件上的修订

如果 E1 通过并走 full 版本:

- **训练侧**:6×A800 在 10M frames 上 5-6 epoch ≈ 5-6 天/run。允许做 ~10 次 ablation 跑(共 ~2 个月训练时间),其余 4 个月用于 debug、E2-E5、写作 ✅ 可行
- **推理侧**:仍受 4090 限制,full 版本的 [C1] **如果坚持用 ViT-L**,需要 H100 推理服务器或者降级到 4090 单步 230 ms(2-4 Hz),**不在控制频率内**
- **建议**:推理 backbone 在 full 版本里也保持 ViT-B(与 MVP 一致),把 [C1] 从 ViT-L 升到 ViT-L 这条改动**单独作为附加 ablation**,要求有 H100 临时使用权才做(workshop 投稿时不卡)

### §0.7.6 Red flag(早期硬件可行性信号)

补充到 §I.4 红旗清单:

7. **第 1 周 [C1] ViT-B + [C9] 1-step inference 在 4090 上实测 > 200 ms**:量化或编译优化没生效。**对策**:用 TensorRT 编译 [C1],用 torch.compile + Triton 优化 [C9],若仍 > 200 ms,backbone 必须降到更小(MobileViT 或 SigLIP-ViT-S)——这会进一步削减基础性能上限,需要重新评估 publishable claim 是否成立。

---

## §A 架构定位与命名

**一句话定位**。CHIME-VLA 与现有 Memory VLA 的根本差别:**(i) 写入控制(when)的监督信号是后视的(hindsight),用 offline saliency 部分替代代理触发器,但部署时仍依赖 self-supervised 预测误差(后视监督在部署时不可得,见 §H Trade-off 1);(ii) 存储路径(how)在容量未满时的 mixing 项全部解析写死(delta-rule + 不可变 bank),容量满后的 consolidation 改为 LRU 丢弃而非合并平均,以保留 delta-rule 的代数性质;(iii) 异质信息(geometric vs semantic)分两个独立通道,每通道有独立的写入触发器与 aux loss**。

**与 v1(CW-Mem)的延续**:不可变 bank、双层放置、后视因果显著性损失。

**与 v1 的差别**:
- v1 的写入触发器是"反事实 surprise"(模型自跑两遍对比),v2 改为 **event-segmentation 预测误差(连续标量)+ HCS offline saliency(离线监督)**双信号。理由:v1 的反事实 surprise 自身需训练、撞 HC-3。
- v1 的"双层放置"是 VLM 端 + action expert 端,v2 改为 **geometric 通道 + semantic 通道**。理由:§3.4 真正的分轴是按精度,不是按位置。
- v1 没有显式的"working buffer / 短期 / 长期"三池区分,v2 引入 **FIFO 工作缓冲(可微)+ 不可变 slot bank(delta-rule + CSM-importance LRU 丢弃)+ 几何 grid(delta-rule + timestamp eviction)** 的明确边界。注意两条容量管理机制不同:slot bank 按 [C12] CSM 重要度排序丢弃,grid 按 [C8] read-timestamp 丢弃 stale voxel——共用"LRU"名称但判据不同。
- v1 §5.1 判断四的"语义信号在子任务边界附近密集 / 感知信号在状态突变附近密集"这条 finer-grained claim,v2 中**简化为同一个 ψ 但两条独立 projection head + 独立 EMA 标准化**(见 [C5])。若该简化不够,fall-back 是把 ψ 拆成 ψ_geo + ψ_sem 两个独立 1-layer Transformer(参数量 +10M)。这是 v2 对 v1 的有意降级,因为"分布密度模式不同"的实证强度不够支持额外 10M 参数。

---

## §B 整体数据流图

```
┌────────────────────── Forward 路径(部署时也跑,< 100 ms 预算) ────────────┐
│                                                                          │
│   o_t (RGB+propio) ──[F1]──► [C1] VLM 主干(frozen + LoRA)                │
│                                       │ h_t ∈ R^{N×d_h}                  │
│                                       │                                  │
│                  ┌───────[F2]─────────┼─────────[F3]──────────┐          │
│                  ▼                    ▼                       ▼          │
│           [C2] FIFO 工作缓冲    [C5] 事件分段触发 ESPC    [C8] 读出接口   │
│              M_work             ─────────────────         (cross-attn)   │
│              (last K_w=8)       γ_geo_t, γ_sem_t ∈ [0,1]                 │
│                  │                    │  sg(γ_geo)            ▲ ▲ ▲     │
│                  │                    ├──────► [C3] Geo 写头 ──┤ │ │     │
│                  │                    │           │            │ │ │     │
│                  │                    │           ▼            │ │ │     │
│                  │                    │     [C6] M_geo grid ───┼─┼─┘[F6] │
│                  │                    │     (multi-res voxel)  │ │       │
│                  │                    │  sg(γ_sem)             │ │       │
│                  │                    └──────► [C4] Sem 写头 ──┤ │       │
│                  │                                │            │ │       │
│                  │                                ▼            │ │       │
│                  │                          [C7] M_sem bank ───┼─┘[F5]   │
│                  │                          (immutable+LRU)    │         │
│                  │                                             │         │
│                  └─────────────[F4]───────────────────────────┘          │
│                                                                          │
│                              [C8] 读出 ──[F7]──► [C9] Action Expert      │
│                                                  (frozen + adapter)      │
│                                                          │ a_t (7-DOF)   │
│                                                          ▼               │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────── Training-only 路径(部署时不跑) ──────────────────────────┐
│                                                                          │
│  [C10] HCS-H:  ∂a*_{t+Δ}/∂o_t  ────►  γ̂_geo_t, γ̂_sem_t (sg before write)│
│                       │                                                  │
│                       └───── L_HCS ─────► [C5] ESPC                      │
│                                                                          │
│  [C11] PRH:    sg(m_t) ── 预测 (o_{t+k}, a*_{t+k}), k∈{4,16,64}           │
│                       │                                                  │
│                       └───── L_PRH ─────► [C3][C4] 写头(经 sg(query))   │
│                                                                          │
│  [C12] CSM:    leave-one-slot-out → KL(π‖π_{-i}) → 槽重要度              │
│                       │                                                  │
│                       └───── L_CSM ─────► [C7] M_sem bank                │
│                                                                          │
│  [C13] Reverse Hindsight Aux:[C10] 在反向 (t→t-Δ) 上算 ∂a*_{t-Δ}/∂o_t   │
│                       │                                                  │
│                       └───── 额外 BCE 标签喂 [C5],无新 loss             │
└──────────────────────────────────────────────────────────────────────────┘

显式 stop-grad(sg)位置(本图全部已标注):

<!-- SG-MATRIX-CANONICAL -->

| # | 位置 | sg 谁 | 理由 |
|---|---|---|---|
| **SG-1** | [C3][C4] 写入瞬间 | sg(γ_*) | L_main 不能流回 [C5] ψ 训写入门——HC-3 |
| **SG-2** | [C8] L_PRH 路径上 query 投影矩阵 | sg(query) | L_PRH 不能经 query → h_t 污染 [C1] LoRA |
| **SG-3** | [C10] 输出 γ̂ 喂 L_HCS 之前 | sg(γ̂) | γ̂ 是 BCE target 不是可学量 |
| **SG-4** | [C12] 经 frozen [C9] | 自然 sg | L_CSM 不会反向训 [C9](否则 [C9] 学到"对 slot 缺失敏感"以 hack L_CSM) |
| **SG-5** | ψ 看到的 M_work 内容(L_HCS path) | sg(M_work) on L_HCS path | L_HCS 不能经 ψ 输入反传到 M_work 内的 h_{t-7..t-1} → [C1] LoRA |
| **SG-6** | [C5] 的 geo_proj / sem_proj | 仅 L_HCS 可训、不接 L_main | 否则 L_main 经预测误差路径污染 ψ |
| **SG-7** | [C8] cross-attn 读端被 L_main 训 | **不能 sg**(结构性) | §6 提到的"被训成短视"风险——L_aux entropy 正则**不是真正的解、是工程兜底**;须监控 attention entropy_min 指标 |

(canonical 来源:proposal §5.4。原文档 §J.1 仅列 SG-1..SG-4 共 4 处,本版本扩展到 7 处,与 proposal 一致。)
```

---

### §B.1 Stop-grad CI 单测契约

每条 SG-N 对应 `tests/test_grad_flow.py::test_sg_<n>_*`,断言 sg-isolated 参数 `param.grad is None`:

| SG | 单测名 | 期望 grad=None 的参数路径 |
|---|---|---|
| SG-1 | `test_sg_1_gamma_to_psi` | `model.heads.espc.psi.*`、`model.heads.espc.geo_proj.*`、`model.heads.espc.sem_proj.*` (when only L_main backwards) |
| SG-2 | `test_sg_2_prh_query_to_perception` | `model.perception.vlm_backbone.*` (when only L_PRH backwards) |
| SG-3 | `test_sg_3_gammahat_target` | `model.training.hcs_consumer.*` (γ̂ tensor `requires_grad=False` 始终) |
| SG-4 | `test_sg_4_csm_through_frozen_action` | `model.action.action_expert.*` (when L_CSM backwards) |
| SG-5 | `test_sg_5_mwork_to_perception_via_psi` | `model.perception.vlm_backbone.*` via `M_work[t-K_w..t-1]` (when only L_HCS backwards) |
| SG-6 | `test_sg_6_proj_only_lhcs` | `model.heads.espc.geo_proj.*`、`sem_proj.*` (when only L_main backwards) |
| SG-7 | `test_sg_7_attention_entropy_floor` | **运行时监控** `H([C8].attn_to_M_work) > entropy_floor`,非 grad assertion |

CI gate:`pytest -x tests/test_grad_flow.py` 必须绿,任何 PR 红就 BLOCK 所有 impl PR(见 PROGRESS.md milestone gate)。

---

### §B.2 单帧 forward 顺序伪代码

定义"一帧 t"内组件执行顺序(消除 [C2] vs [C5] vs [C3]/[C4] 的并发歧义):

```python
# Forward path (deploy + train, < 100 ms budget)
h_t = C1(o_t, proprio_t)                            # [B, N=256, d_h=1152]
gamma_geo, gamma_sem = C5(h_t, M_work)              # ESPC reads M_work^{t-1} (BEFORE append)
M_work = C2.append(M_work, h_t)                     # FIFO ring update
C3.write_(h_t, sg(gamma_geo), M_geo)                # geo write head, in-place delta-rule
C4.write_(h_t, sg(gamma_sem), M_sem)                # sem write head, slot_free mask applied
c_t = C8(M_work, M_geo, M_sem, h_t)                 # read interface: cross-attn + trilinear
a_t = C9(c_t, h_t)                                  # action expert (flow matching, 1-step distill MVP)

# Training-only (after T steps in episode)
gamma_hat_geo, gamma_hat_sem = HindsightConsumer.load(episode_id)   # [C10] offline product
L_HCS = BCE(gamma_geo_seq, sg(gamma_hat_geo)) + BCE(gamma_sem_seq, sg(gamma_hat_sem))
L_PRH = sum_k MSE(C11(sg(m_t)), (o_{t+k}, a*_{t+k})) for k in {4,16,64}
L_CSM = csm_leave_one_out(C12, m_t_seq, M_sem, frozen_C9, n_slots=4)
L_aux = -lambda_ent * H(C8.attn_weights_to_M_work)
L_main = flow_match(a_seq, a*_seq)
L_total = L_main + lambda_1(step) * L_HCS + lambda_2 * L_PRH + lambda_3 * L_CSM + lambda_4 * L_GC + L_aux
```

**关键不变量**:
- C5 在 C2.append **之前**调用,以保 ψ 看到 `M_work^{t-1}`(predict h_t 的过去窗口)
- C3/C4 在同一时间步可并行(无相互依赖);代码上可同步串行执行,语义不变
- L_PRH 的 m_t 输入必须 `sg(m_t)`(与 C8 query sg 形成两道隔离)
- L_HCS 仅训 [C5] {ψ, geo_proj, sem_proj}——其余参数对 L_HCS path 全部 sg(SG-5/SG-6 联立)

---

## §C 组件卡片(每张可独立阅读)

> **阅读顺序提示**:卡片相互独立,依赖以"前置依赖"行显式标注。如果只读两张,推荐 [C5] ESPC 与 [C10] HCS-H —— 它们承载本架构与现有 Memory VLA 工作的核心差异。

---

### [C1] VLM 主干(Perception Backbone)

**前置依赖**:无。

**解决约束**:无新增,提供 h_t 给所有下游。

**接口契约**:
- 输入:o_t = (RGB ∈ R^{H×W×3}, proprio ∈ R^7),频率 5–10 Hz
- 输出:h_t ∈ R^{N×d_h},N≈256 token,d_h=1152(SigLIP-ViT-L 默认)
- 频率:与控制频率同步,每步触发

**内部 operator**:SigLIP-ViT-L(或 π0.5 encoder),全部参数 frozen;在 cross-attn 输出端挂秩 r=16 的 LoRA,只这部分可训。

**最近原型**:π0.5 encoder。差别:加 LoRA 而非全参微调,堵窄主任务 loss 进主干的污染。

**可学性 / 参数量**:LoRA ~5M 可训,主干 ~600M 冻结。

**梯度路径**:L_main → LoRA(主干 frozen 部分梯度截断)。L_PRH 因 [C8] query sg 不流入 [C1]。

**可独立验证性**:不必。

---

### [C2] FIFO 工作缓冲 M_work

**前置依赖**:[C1] 提供 h_t。

**解决约束**:HC-1(短期容量)、HC-2(短期梯度可达 trivially)。**不豁免** HC-3:见下方梯度路径段。

**接口契约**:
- 输入:每步 [F2] 来的 h_t,append-only ring buffer
- 输出:M_work ∈ R^{K_w×d_h},K_w=8 [标定值,需小规模实验确认]

**内部 operator**:
$$M_{\text{work}}^{(t)} = \text{concat}(M_{\text{work}}^{(t-1)}[1:], h_t)$$

**最近原型**:Diffusion Policy with 8-frame history(Chi et al. 2023)。差别:无。

**可学性**:无可训参数。

**梯度路径**:L_main 沿 [C8] cross-attn 反传到 M_work 中各 h,再沿 LoRA 到主干 LoRA。
**风险**:cross-attn 端可学,L_main 可能训出"只 attend 最新一帧"的短视读出 → 等价于 §2.1 短视失败模式被推到 read 端。
**对策**:在 [C8] cross-attn 上加 attention entropy 正则(λ_ent=0.01,见 §D L_aux),若 ablation 显示无 collapse 可关闭。

**可独立验证性**:不必。

---

### [C3] 几何写头(Geometric Write Head)

**前置依赖**:[C1] h_t、[C5] γ_geo_t。**前置 [C5] 未就绪时**,可用 e_geo_t 标量(通过帧间余弦相似度自计算)做 stub 跑 sanity check。

**解决约束**:HC-7、HC-3(γ_geo 在写入时被 sg,L_main 不流入 [C5])。

**接口契约**:
- 输入:h_t ∈ R^{N×d_h},sg(γ_geo_t) ∈ R(标量,来自 [C5])
- 输出:增量 ΔM_geo,写入 [C6] 的多分辨率 voxel grid
- 频率:每步

**内部 operator**:
1. 投影 MLP token_to_voxel:对每个 token,预测 (x,y,z) ∈ [0,1]³,得 N 个 anchor 点。
2. 三个分辨率层(coarse 8³、mid 16³、fine 32³),三线性插值逆向写入:
   $$\Delta M_{\text{geo}}^{(\ell)}[\text{voxels near } x_i] \mathrel{+}= \mathrm{sg}(\gamma_{\text{geo}}^{(t)}) \cdot \alpha_\ell \cdot \phi_\ell(h_t^{(i)})$$
   $\alpha_\ell$ frozen 常数(coarse:0.5, mid:0.3, fine:0.2)。
3. **delta-rule**:无门控乘子,sg(γ_geo) 是 0/1 软门;grid 容量满后 **timestamp eviction** 淘汰最早未读条目(由 [C8] 读时盖 timestamp 维护)。注意:这与 [C7] slot bank 的"CSM-importance LRU"是两套不同算法,共用"丢弃"语义但判据不同。

**最近原型**:NICE-SLAM(Zhu et al. 2022)multi-resolution feature grid。差别:无 RGB-D 监督,内容由 BC + L_PRH 反向定义。

**可学性 / 参数量**:投影 MLP ~2M。Grid 是 buffer。

**梯度路径**:L_main 经 [C8] read → 三线性反向 → grid → φ_ℓ → h_t。注意 sg(γ_geo) 阻断梯度从 [C3] 流入 [C5]。L_PRH 经 sg(query) 同路径。

**可独立验证性**:**可独立验证**(配 stub [C5])。最小验证实验:RoboCasa 单子任务"取物体 A",固定 [C5][C6][C9],只训 [C3] + L_main + L_PRH;对比"用 grid 读出 vs 不读"在精确抓取的 SR。1k demos,1×H100,12 小时。

---

#### [C3] 补充:几何写头的完整逻辑

##### A. 一段话讲清楚

**[C3] 在做什么**:每帧拿到 [C1] VLM 输出的 hidden state h_t(256 个 token,每个 1152 维),把其中"任务相关的几何信息"按 **3D 物理位置**索引存到一张固定大小的 **3D 画布** [C6] M_geo 上。这张画布跨整个 episode 持续累积,不被时间衰减——belief state 跨遮挡、跨百步保留都靠它。

**怎么做的**:让每个 token 自己说"我属于工作空间的哪个 3D 位置"(token_to_voxel MLP 投影),然后在那个位置上对应的 voxel 周围"加一笔" feature(纯加性 scatter)。每帧最多动 ~2048 个 voxel,其他 30000+ voxel 字面意义上一动不动。**没人动的 voxel = 完全不变**——这就是"零衰减"的代数本质,与门控 RNN 全局乘 (1-g) 的根本差别。

**关键反直觉点**:M_geo 就是一个 R^{42000×64} 的固定 tensor,**记忆的物理实体就是这个 tensor 当前的数值**——没有过去帧 list、没有事件序列、没有时间戳。所有跨长程几何信息都"塌"在这个 tensor 的当前数值里。

**与 [C4] slot bank 的分工**:画布按 **3D 物理坐标**索引,装"哪个位置有什么东西";slot bank 按 **content-based key** 索引,装"发生过什么事件"。两者互补回应 HC-7 与 §3.4——精度异质 = 索引异质 = 双通道。

---

##### B. 物理结构:画布长什么样

工作空间(机器人能触及的 ~1m³ 立方体)被切成三层不同分辨率的 voxel grid:

```
M_geo_coarse ∈ R^{8 × 8 × 8 × 64}     每格 ≈ 12.5cm  (512 voxel)
M_geo_mid    ∈ R^{16 × 16 × 16 × 64}  每格 ≈ 6.25cm  (4096 voxel)
M_geo_fine   ∈ R^{32 × 32 × 32 × 64}  每格 ≈ 3.1cm   (32768 voxel)
```

三层共 ~4.2 万个 voxel,每个装 64 维向量。Episode 开始时**全 0**——无任何记忆。

**三层各自承担什么**:
- coarse 装"场景级布局"——哪一大块区域有东西,梯度密集、收敛快
- mid 装"物体级位置"——物体落在哪个中块格子
- fine 装"亚厘米精度"——物体的精确位姿,稀疏命中

**voxel 不是物体、不是过去帧、不是事件**——它就是"3D 空间某个 3cm³ 小格上累积了什么 feature"。这是与 slot bank 的根本区别:

| | [C6] M_geo voxel grid | [C7] M_sem slot bank |
|---|---|---|
| 索引轴 | **3D 物理坐标** (x,y,z) | **content-based key**(语义相似度) |
| 一格装什么 | "这个 3D 位置的几何 feature" | "某个事件的语义 feature" |
| 适合存什么 | "红块在 (0.5, 0.5, 0.3)" | "红块入抽屉这件事已发生" |
| 写入触发 | γ_geo(几何状态突变) | γ_sem(子任务边界) |

---

##### C. 写入流程:每帧四步

**桥梁问题先讲清楚**:VLM 输出 256 个 token,每个 token 是按 ViT 序列 1D 索引排列的 hidden state——**它不直接编码 3D 空间位置**。所以需要一个软投影 module 把"token 的语义"映射到"3D 物理位置",这就是 `token_to_voxel` MLP 的角色。它通过 BC + L_PRH 间接监督训练而成,**无 RGB-D 显式监督**——所以学到的是"任务相关的 3D 抽象投影",不是几何重建。

每帧 t,[C3] 对每个 token 做四步:

**Step 1 — 算 3D anchor 坐标**
```
x_i = token_to_voxel_MLP(h_t^{(i)})  ∈  [0,1]³
```

例:t=60 时 token 17 编码"红块"语义 → MLP 投到 (0.5, 0.5, 0.3);token 23 编码"机械手"→ 投到 (0.6, 0.6, 0.5);token 0 编码"背景墙"→ MLP 学会投到 grid 边界外或某个废弃 voxel(任务无关 token 自然被压缩到不污染容量的位置,这是 emergent 行为)。

**Step 2 — 三线性插值找 8 个邻居 + 权重**
x_i = (0.5, 0.5, 0.3) 在 fine 层对应连续坐标 (16, 16, 9),周围 8 个 voxel 是 (16,16,9)..(17,17,10),三线性权重 w_1..w_8(按距离反比,和=1)。

**Step 3 — 算这个 token 在 voxel 空间的 feature**
```
f_i = φ_ℓ(h_t^{(i)})  ∈  R^{64}     ← 1152 → 64 可学投影,每层 ℓ 一个
```

**Step 4 — 加性 sparse scatter(三层同时写)**
```python
for ℓ in [coarse, mid, fine]:
    for j in range(8):  # 8 个邻居 voxel
        M_geo[ℓ][neighbor_j] += sg(γ_geo) · α_ℓ · w_j · f_i
```

α_ℓ frozen 常数(coarse 0.5, mid 0.3, fine 0.2)——**任何与 mixing 相关的乘子都不可学是 HC-3 的强制要求**,否则 truncated BPTT 会训坏。

**关键不变量**:这个 += 只对 8 个邻居 voxel 执行,**其他 voxel 字面意义上一动不动**——这是后面所有性质的代数基础。

---

##### D. 跨长程记忆保留:"加 0 = 不变"是字面意思

抽屉任务里,假设机械手 t=10 到 t=30 持续在 (0.3, 0.3, 0.4):

```
t=10:  voxel (9,9,12) 从 0 累积到 ~0.05
t=15:  累积到 ~0.4
t=30:  累积到 ~1.0     ← 多帧融合,信号增强
t=31:  机械手移走,no token 投到 (9,9,12) → voxel += 0
t=32:  voxel += 0
...
t=200: voxel 仍 ≈ 1.0   ← 跨 170 步零衰减
```

**voxel (9,9,12) 在 t=31 后不变,不是因为有保护机制,而是没人动它**——`token_to_voxel` MLP 没把任何 token 投到那个位置,所以 `M_geo[(9,9,12)] += ...` 那一行代码这一帧根本没执行。

对比门控 RNN(类 B 工作的代数本质):
```
门控更新:    M = g·new + (1-g)·M     ← 对所有 voxel 无差别施加
             每帧都乘 (1-g) → 几何衰减,§3.2 不可避免

画布 sparse: M[hit] += new            ← 只对命中位置加,其他不动
             无 (1-g) 乘子 → 零衰减
```

衰减需要"某个动作把内容稀释",画布架构里**那个动作不存在**——这就是 §3.2 不发生的具体机制。

---

##### E. 写入是 sparse 的:机械臂动多少都不会写满

直觉上"机械臂动很多 → 经过很多 3D 位置 → grid 应该被写满"是错的。三层原因:

1. **写入位置由 token 内容决定,不是机械臂物理位置**。`token_to_voxel` MLP 看的是 hidden state 表征,不是相机几何。机械臂在哪儿物理上**不直接**决定写哪个 voxel——机械臂的 token 投到对应位置,但**只有视野中的任务相关物体**有 token,场景里其他位置没有 token 写它。

2. **每帧写入数量有固定上限**。256 token × 8 邻居 = 最多 2048 voxel / 帧(实际更少,~500-1500,因为重叠)。机械臂动 1 步、100 步、1000 步,**每帧上限不变**——差别只在"写到哪些 voxel",不在"写多少"。

3. **背景 token 被 MLP 学会投到 grid 之外**。L_PRH 监督训 MLP:把背景 token 投到任务区域会污染 voxel feature → m_t 预测能力下降 → 梯度反传修正 MLP。久了 MLP 学会忽略背景——投到 grid 边界外或聚集到某个废弃 voxel,**容量被任务相关内容占据**。

**结论**:典型 200-step 任务,实际被写入的 voxel 占总容量 1-10%,**90%+ voxel 长期为 0**,LRU 丢弃几乎不触发。

---

##### F. 多次写入同一 voxel 的三种结果

画布架构里**只有三种操作**会改变 voxel 内容,**没有第四种**(自然衰减、被新观测覆盖、随时间淡化都不发生):

| 操作 | 何时发生 | 结果 |
|---|---|---|
| **累加(同物体反复)** | 同一物体在同一 3D 位置出现多帧 | feature 多帧融合,信号增强,噪声抵消,信噪比提高 |
| **叠加(不同物体共位)** | 不同物体先后路过同一物理位置(罕见) | 两个 feature 向量加和,read 时变模糊但都部分保留 |
| **LRU 整体清零** | grid 容量阈值触发(<10% 任务下几乎不发生) | 整个 voxel 区域置零,显式动作 |

注意:**"取代"这个操作根本不存在**——没有"新内容覆盖旧内容"这条 code path。

---

##### G. 跨遮挡 belief 保持 + 完整生命周期

这是画布架构的杀手锏——belief state 跨遮挡保留,纯靠加性更新的代数性质完成,不依赖任何长程 BPTT、不依赖事件分类器、不依赖 VLM 慢推理。

```
─────────────────────────────────────────────────────────
t=0       M_geo 全 0
─────────────────────────────────────────────────────────
t=1-9     起手阶段,γ_geo 中等
          ~1500 voxel 从 0 变 ~0.05 量级
          30k+ voxel 仍 0
─────────────────────────────────────────────────────────
t=10-30   机械手在 (0.3, 0.3, 0.4) 持续操作
          voxel (9,9,12) 累积到 ~1.0(多帧融合强化)
─────────────────────────────────────────────────────────
t=31-59   机械手移到别处
          voxel (9,9,12) 每帧 += 0 → 内容不变
          新位置 voxel 开始累积
─────────────────────────────────────────────────────────
t=60      红块入抽屉,γ_geo ≈ 0.7
          fine voxel(抽屉内位置)强写入到 ~2.0
─────────────────────────────────────────────────────────
t=61-89   红块在抽屉内仍可见
          抽屉内 voxel 继续累积到 ~3.0
─────────────────────────────────────────────────────────
t=90-100  抽屉关闭,红块完全遮挡
          红块 token 不再出现 → voxel(抽屉内) += 0
          内容保持 ~3.0  ← 跨遮挡 belief 保留
─────────────────────────────────────────────────────────
t=95      [C9] 决策"是否重开看红块?"
          [C8] 读 voxel(抽屉内) → feature ≈ 3.0
          action expert 知道"那位置有东西" → 不重开,继续关
─────────────────────────────────────────────────────────
t=100     episode 结束,M_geo 当前数值 = 整段任务的几何记忆
─────────────────────────────────────────────────────────
t=101     下一 episode,M_geo 重置全 0(放弃跨 episode 学习)
─────────────────────────────────────────────────────────
```

---

##### H. 时间信息:画布显式不承担

画布不存"何时被写入"的时间戳——这是有意的。如果 voxel 装时间,等于退化回时序记忆,§3.2 几何衰减重新发生。架构按以下分工:

| 时间信号类型 | 承担组件 | 机制 |
|---|---|---|
| 短期精确顺序(< 8 步) | [C2] FIFO M_work | 8 帧 ring buffer,顺序明确 |
| 中期事件序(子任务边界) | [C7] M_sem slot bank | slot 写入序 = 事件发生序 |
| 长期空间状态(物体在哪) | [C6] M_geo 画布 | 空间索引,无显式时间 |

画布唯一的隐式时间信号:**feature 量级反映"出现频率"**(累积幅度高 = 反复出现 = 高可信)——但"何时出现"答不出,要靠 [C2] / [C7] 互补。

---

##### I. 一句话总结

[C3] 写入 = **每帧每个 token 通过 token_to_voxel MLP 投到 3D 位置,在对应 voxel 周围 8 邻居做"加 sg(γ_geo)·α·w·feature"的纯加性 sparse scatter**。**没命中的 voxel 完全不被触碰**——这是"加 0 = 不变"的字面意思,也是与门控 RNN 全局乘 (1-g) 的代数差别。机械臂动多少都不会写满 grid,因为写入位置由 token 内容驱动而非物理覆盖,且 MLP 学会忽略背景。**记忆的物理实体 = M_geo 这个 tensor 当前的数值**,通过 sparse 加性累积和零衰减性质实现跨长程几何状态保存。它不承担时间信息——时间维度由 [C2] FIFO 与 [C7] slot bank 互补承担,这是 HC-7 与 §3.4 双通道设计的几何半边。

---

### [C4] 语义写头(Semantic Write Head)

**前置依赖**:[C1] h_t、[C5] γ_sem_t、[C7] slot bank。**前置 [C5] 未就绪时**,用 frame-level 语义相似度突变阈值做 stub。

**解决约束**:HC-7、HC-2(slot 分配 content-based,可微)、HC-3(γ_sem 写入时 sg)。

**接口契约**:
- 输入:h_t,sg(γ_sem_t)
- 输出:更新 [C7] M_sem 中按 softmax 分布的 slot

**内部 operator**:
1. q_t = MLP_q(pool(h_t)) ∈ R^{d_s}
2. $s_t^{(i)} = \cos(q_t, k_i)$
3. $w_t^{(i)} = \text{softmax}(s_t / \tau)$,τ=0.5 [标定值]
4. **Delta-rule 写入**:
   $$M_{\text{sem}}^{(t),i} \leftarrow M_{\text{sem}}^{(t-1),i} + \mathrm{sg}(\gamma_{\text{sem}}^{(t)}) \cdot w_t^{(i)} \cdot \phi(q_t) \otimes v_t$$
   v_t = MLP_v(pool(h_t)),φ = ELU+1。**关键**:加性更新,无 (1-g) 衰减乘子。
5. **容量管理**:bank 满时(K_s=64 全部 ‖v_i‖ > 阈值),触发 **LRU 丢弃**(基于 [C12] CSM 算的最低重要度 slot):丢弃 w_i^{CSM}最低的 1 个 slot 整体置零(value + key)。**这条与原草案的"相似度 > 0.9 合并"不同 ——合并平均会引入 g=0.5 mixing,破坏 §3.2 代数性质;LRU 丢弃保留 delta-rule 的全部代数证明。代价:被丢弃的 slot 内容彻底丢失。该 trade-off 在 §H Trade-off 5 显式承认。**

**最近原型**:Infini-attention(Munkhdalai 2024)delta-rule + Slot Attention softmax + XMem importance-based 替换。差别:删合并平均,改 LRU 丢弃。

**可学性 / 参数量**:MLP_q + MLP_v ~3M。

**梯度路径**:L_main → [C8] read → softmax_i → q_t, v_t → MLP → h_t。
- sg(γ_sem) 阻断 L_main 流入 [C5]。
- L_CSM 流入 MLP_q/MLP_v。
- L_PRH 经 sg(query) 流入 MLP。

**可独立验证性**:**必须和 [C5][C7][C10] 联立验证**。

---

#### [C4] 补充:语义写头的完整逻辑

##### A. 一段话讲清楚

**[C4] 在做什么**:每帧拿到 [C1] VLM 输出的 hidden state h_t(256 个 token,经过视觉-语言-本体跨模态融合),把其中**事件级语义**(抽屉打开了、红块入了、第一阶段完成了)按 **content-based key** 索引存到一个 **64 槽的事件 bank** [C7] M_sem 里。任务后期决策需要回忆"我开过抽屉吗"、"红块放进去了吗",都靠它。

**怎么做的**:核心是两件事——**(a) 把这一帧的整体语义压成一个"事件指纹" q_t,(b) 把这个指纹路由到 64 个 slot 中最像它的那个 slot 上,做纯加性累加**。具体四步:

1. `pool(h_t)` 把 256 个 token 压成 1152 维全局向量(整帧整体语义)
2. `q_t = MLP_q(pool(h_t))` 投到 256 维 slot 路由空间(事件指纹)
3. q_t 与 64 个 **frozen 随机 key** k_1..k_64 算余弦相似度,softmax 出权重 w_1..w_64(典型 1-2 个 slot 拿走 ~0.7,其他几乎为 0)
4. 在每个 slot value 上加性写入:`v_i += sg(γ_sem) · w_i · φ(q_t) ⊗ v_t`,γ_sem 由 [C5] 软门控

**关键反直觉点**:64 个 slot key 是 **frozen 的随机向量**,**不携带任何预设语义**。slot #5 装"抽屉打开"完全是因为 t=30 时 q_30 恰好与随机的 k_5 余弦相似度最高——**slot 的语义是 emergent 的,由 first-write 事件随机绑定**,后续相似事件因 q 与 k_5 仍最相似而继续路由到 slot #5,信号累加强化。这种"随机绑定 + 相似度强化"等价于**可微版的哈希表**:保 HC-2 全可微,同时规避 HC-3 的可学 mixing 问题。

**与 [C3] 画布的分工**:画布按 **3D 物理坐标**存"哪儿有什么"(空间索引),slot bank 按 **content-based key** 存"发生过什么事件"(语义索引)。两者互补回应 HC-7 与 §3.4——精度异质 = 索引异质 = 双通道。

---

##### B. 物理结构:slot bank 长什么样

```
M_sem 由 64 个 slot 组成,每个 slot 有两个组件:
  k_i ∈ R^{256}    "地址"——episode 开始时随机初始化 + frozen,不变
  v_i ∈ R^{256}    "内容"——从 0 开始,通过 [C4] delta-rule 累加
```

Episode 开始:k_1..k_64 随机方向(无语义)、v_1..v_64 全 0(无记忆)。

**slot 不是 3D 位置,slot 是一个抽象事件槽**。slot #5 不对应工作空间的某个角落;slot #5 是"64 个互相独立可寻址槽里的第 5 个"。它装什么由 episode 中第一个被路由到它的事件决定——内容是 emergent 的。

与 [C3] voxel 画布的根本区别:

| | [C6] M_geo voxel grid | [C7] M_sem slot bank |
|---|---|---|
| 索引方式 | **3D 物理坐标** (x,y,z) | **content-based key**(语义相似度) |
| 一格装什么 | "这个 3D 位置的几何 feature" | "某个事件的语义 feature" |
| 写入触发 | γ_geo 高(几何状态突变) | γ_sem 高(子任务边界) |
| 写入位置由谁决定 | token_to_voxel MLP 投到哪 | q_t 与哪个 slot key 余弦相似度最高 |
| 适合存什么 | "红块在 (0.5, 0.5, 0.3)" | "红块入抽屉这件事已发生" |
| 维度 | d_g=64(局部几何熵低) | d_s=256(事件语义熵高) |

---

##### C. 为什么 key 是 frozen 随机向量

这是设计的核心反直觉点。直觉上你会问:"key 不应该学到表示某种语义吗?" 答案是不学,而且是有意不学。

**如果 key 可学**,L_main 训练会让 k_i 朝"对决策有用"的方向漂移。问题是 slot 索引稳定性会被破坏:t=10 写入 slot #5 时 k_5 是某个方向,t=80 想读 slot #5 时 k_5 已经飘到另一个方向——**两次访问的"地址"指向了不同的语义内容**,记忆错位。

**frozen 随机 key 等价于哈希表的桶号**:每个 slot 是一个固定地址,**地址本身不携带语义,语义在 value 里、由内容决定**。哈希表的妙处不是"桶号长什么样",而是"不同事件被路由到不同桶"——本架构的 slot 用同样的逻辑。

**emergent 语义机制**:
```
Episode 开始:
  k_5 = (随机方向 R_5)  ← 无语义
  v_5 = 0

t=30:  q_30(代表"抽屉打开")与 64 个 k 算 cosine
       恰好 k_5 余弦相似度最高(纯偶然)
       → softmax w_5 ≈ 0.7
       → v_5 += 0.85 · 0.7 · φ(q_30) ⊗ v_30
       slot #5 从此装上"抽屉打开"语义

t=后续:  类似事件出现 → q 仍与 k_5 最相似(因为 k_5 不变)
        → 继续路由到 slot #5,累加强化
        → slot #5 越来越"专门"装抽屉打开
```

frozen 随机 key 让 slot 的语义身份**靠 first-write 随机绑定 + 相似度自然保持**——这是规避 HC-3 的关键(任何与 mixing/路由相关的可学参数都会被 truncated BPTT 训坏)。

---

##### D. 写入流程:每帧四步

每帧 t,[C4] 做这四步:

**Step 1 — 算事件指纹**
```
pool(h_t) = mean(h_t, dim=0)  或  h_t[CLS]   ∈ R^{1152}
q_t = MLP_q(pool(h_t))                        ∈ R^{256}
```
pool 把 256 个 token 压成全局向量(代表整帧状态),MLP_q 投到 slot 路由空间。

**Step 2 — 与 64 个 frozen key 算余弦相似度**
```
s_t^{(i)} = cos(q_t, k_i)        i = 1..64
```

**Step 3 — Softmax-over-slots 竞争分配**
```
w_t^{(i)} = softmax(s_t / τ)     τ = 0.5
```
最相似的 slot 拿走大部分权重(~0.7),次相似的少量(~0.1),其他几乎 0。**关键:soft 加权而非 hard top-1**——保 HC-2 全可微。

**Step 4 — Delta-rule 加性写入**
```
v_t = MLP_v(pool(h_t))    ∈ R^{256}     ← 写入内容,与 q_t 平行的另一个 MLP

对每个 slot i:
  v_i += sg(γ_sem) · w_t^{(i)} · φ(q_t) ⊗ v_t
                                ↑外积
                                φ = ELU+1
```

**MLP_q 与 MLP_v 的分工**:MLP_q 算"路由地址",决定写到哪个 slot;MLP_v 算"具体内容",决定写入的 feature。两者目标不同(区分性 vs 完整性),所以分开。类比邮件:MLP_q 是分类标签,MLP_v 是邮件内容,k_i 是文件夹位置编号,v_i 是文件夹累积的邮件。

**关键性质**:加性更新,**无 (1-g) 衰减乘子**——与 [C3] 同样的代数性质,跨 200 步内容仍可读。sg(γ_sem) 阻断 L_main 训写入门(HC-3)。

---

##### E. 不会被覆盖的代数原理(数值演示)

直觉上你会问:"t=60 时 q_60 与 k_5 也可能有点相似,slot #5 不会被'红块入抽屉'的内容污染吗?"

**会有微量污染,但不会被覆盖**。具体演算:

```
t=30 写入"抽屉打开"后:
  v_5 = 0.85 · 0.7 · φ(q_30) ⊗ v_30
       ≈ 量级 0.6   ← slot #5 主成分

t=60 写入"红块入抽屉":
  q_60 与 k_5 余弦相似度 ≈ 0.1(因为 k_5 已经"绑定"到 q_30 方向)
  softmax 后 w_5 ≈ 0.05
  v_5 += 0.85 · 0.05 · φ(q_60) ⊗ v_60
       ≈ 量级 0.04   ← 微量污染,加性

t=60 写入后:
  v_5 = (主要"抽屉打开" feature) + (微量"红块入抽屉" feature)
      ≈ 量级 0.6 ± 0.04
```

**slot #5 仍然主要装"抽屉打开"**——后续写入的 w_5 都很小(因为 q 已经分到别的 slot 了),加性扰动微弱。read 时 cross-attn 回归到 slot #5 的主成分,微量扰动作为噪声被掩盖。

这是 softmax-over-slots 的核心好处:**相似事件分到同一 slot 累加(强化),不相似事件分到不同 slot(隔离)**——天然实现"事件聚类",**不需要任何显式聚类机制**。

---

##### F. 抽屉任务的完整时序演示

```
─────────────────────────────────────────────────────────
t=0     M_sem 全 0, k_1..k_64 随机初始化 frozen
─────────────────────────────────────────────────────────
t=15    伸手中,γ_sem ≈ 0.1(子任务无突变),几乎不写入
        即便 q_15 与某 k_i 相似,sg(γ_sem)·w 也太小
─────────────────────────────────────────────────────────
t=30    "抽屉打开"瞬间,γ_sem ≈ 0.85
        q_30 = MLP_q(pool(h_30)) 代表"抽屉刚被开"
        与 64 个 k 算相似度 → 假设 k_5 最相似
        softmax w_5 ≈ 0.7
        v_5 += 0.85 · 0.7 · φ(q_30) ⊗ v_30
        ─→ slot #5 装上"抽屉打开事件"
─────────────────────────────────────────────────────────
t=60    "红块入抽屉"瞬间,γ_sem ≈ 0.85
        q_60 代表"红块入抽屉"
        与 k_5 不太相似(那是抽屉打开方向)
        与空 k_18 最相似(余弦近随机)→ w_18 ≈ 0.7
        v_18 += 0.85 · 0.7 · φ(q_60) ⊗ v_60
        ─→ slot #18 装上"红块入抽屉"
─────────────────────────────────────────────────────────
t=85    "抽屉关到一半",γ_sem ≈ 0.6
        q_85 与已写入 slot 都不太像,与某空 k_27 最相似
        v_27 += 0.6 · 0.6 · φ(q_85) ⊗ v_85
        ─→ slot #27 装上"关闭进行中"
─────────────────────────────────────────────────────────
t=95    [C9] 决策"是否重开看红块?"
        [C8] 读 M_sem,query 与 k_18 相似度仍高
        cross-attn 命中 v_18 → "红块已入抽屉"语义
        ─→ action expert 决策:不重开
─────────────────────────────────────────────────────────
t=100   episode 结束
        非零 slot:#5(开抽屉)、#18(放红块)、#27(关抽屉)
        其余 61 个 slot 仍 ≈ 0
─────────────────────────────────────────────────────────
t=101   下一 episode,M_sem 全部清零(放弃跨 episode)
─────────────────────────────────────────────────────────
```

**关键观察**:整个 200-step 任务只用了 3-5 个 slot,61 个 slot 长期为 0。这是因为(a) 真正的语义事件就 3-5 件、(b) γ_sem 大多数帧 < 0.2 实际写入幅度极小、(c) softmax 把每次写入集中到 1-2 个 slot 不弥散。**K_s=64 对 200 步任务远超够用**(§H Trade-off 4 提到 >500 步时才可能不够)。

---

##### G. 失去内容的三种方式

slot 与 voxel 一致,**只有三种丢失方式**(自然衰减、被新观测覆盖、随时间淡化都不发生):

| 丢失方式 | 何时发生 | 主动/被动 |
|---|---|---|
| **LRU 整体清零** | bank 满时丢弃 [C12] CSM 重要度最低的 slot,value + key 整体置零 | **主动**(显式动作) |
| **Hash collision** | 多个事件 q 与同一 k_i 高相似时共写,内容向量加和 | **被动但不删除**(混合,read 时变模糊) |
| **Episode 结束** | 整个 bank 重置 | **主动** |

LRU 用 CSM 重要度而不是 last-read time,因为 slot bank 的容量比 voxel grid 紧——必须丢"对决策最不重要"的而不是"最久没读"的。**LRU 不是合并平均**(那会引入 g=0.5 mixing 破坏 §3.2 代数性质),而是整个 slot value + key 同时清零,腾出位置给新事件随机绑定。

---

##### H. 与画布的双通道协同

抽屉任务里两边各装什么、决策时怎么校验:

| 时刻 | [C6] M_geo voxel grid | [C7] M_sem slot bank |
|---|---|---|
| t=30 抽屉打开 | voxel(抽屉位置)累积"那里有打开的抽屉" feature | slot #5 装"抽屉打开"事件语义 |
| t=60 红块入抽屉 | voxel(抽屉内位置)累积"那里有红块" feature | slot #18 装"红块入抽屉"事件语义 |
| t=90 关抽屉,红块遮挡 | voxel(抽屉内)仍 ≈ 红块 feature(不变) | slot #27 装"关闭进行中" |
| t=95 决策"重开吗?" | 读 voxel(抽屉内)→ "那位置有东西" | 读 slot #18 → "已经放过红块" |
| **决策结果** | **两边匹配 → 高置信"不重开,继续关到底"** | |

**双向校验是核心好处**:几何画布说"位置有东西",slot bank 说"事件已发生"——两者匹配 = 高置信,两者矛盾(画布说有、slot 说没)= 可能 false positive 或前一帧 hash collision,低置信。这是 HC-7 / §3.4 双通道设计的语义半边。

---

##### I. 一句话总结

[C4] 写入 = **每帧通过 q_t 与 64 个 frozen 随机 key 的余弦相似度做 softmax-over-slots,把当前 hidden state 的语义 feature 加性累积到最相似的 1-2 个 slot 上**,sg(γ_sem) 控制写入幅度。slot 装的是"事件级语义"——抽屉开过、红块入了、抽屉关了——而非位置级几何。**slot 语义由 first-write 事件随机绑定**(等价于可微哈希表),不被覆盖、只被累加或微量污染。失去内容只发生在 LRU 显式丢弃、hash collision 混合、episode 重置三种情况。与 [C3]/[C6] 几何画布配对使用,**几何走空间索引、语义走 content-key 索引**——两者互补回应 HC-7 与 §3.4 的精度异质性,这是双通道设计的语义半边。

---

### [C5] 事件分段触发器(ESPC)

**前置依赖**:[C1] h_t、[C2] M_work^(t-1)。

**解决约束**:HC-5(部分,见 §H Trade-off 6)、§3.1 部分(代理信号 + 后视监督双层)。

**接口契约**:
- 输入:h_t,M_work^(t-1)
- 输出:γ_geo_t ∈ [0,1],γ_sem_t ∈ [0,1]
- 频率:每步,**5-15 ms 在 H100 上**(纸面估算口径,需实测确认;详见 §0.7 与 §H Trade-off 3)

**内部 operator**:
1. **预测下一帧 hidden**:1-layer Transformer ψ,输入 M_work^(t-1) 全部 K_w·N=2048 token,输出 ĥ_t|t-1。
2. **分通道预测误差**:
   - $e_{\text{geo}}^{(t)} = \|\text{geo\_proj}(h_t) - \text{geo\_proj}(\hat h_{t|t-1})\|_2$  (token 局部投影,d_proj=64)
   - $e_{\text{sem}}^{(t)} = 1 - \cos(\text{sem\_proj}(\text{pool}(h_t)), \text{sem\_proj}(\text{pool}(\hat h_{t|t-1})))$
3. **EMA 标准化**:γ_*_t = sigmoid((e_*_t − μ_*) / σ_*),μ_* / σ_* 用全数据 EMA 维护(frozen,非可学,EMA 系数 0.99)。
4. **HCS 监督(训练时)**:[C10] 给出 γ̂_*_t 作为 BCE 目标。

**最近原型**:Baldassano et al. 2017(神经科学 event segmentation)。差别:Baldassano 是观测者 fMRI 模型,CHIME 把同一 prediction error 思想搬到 control backbone。**[原型在 ML 文献内为新设计],我承认这是 v2 内部新颖性最高、原型最弱的组件之一。**

**可学性 / 参数量**:ψ ~10M,geo_proj/sem_proj ~1M。

**梯度路径**:
- **L_main 不流入 ESPC**,因为 sg(γ_*) 在 [C3][C4] 写入瞬间。
- **L_HCS → ψ + projections**(sg γ̂)。
- **L_PRH 不流入 ESPC**(stop-grad on γ_* in [C10]'s gradient path)。

**可独立验证性**:**部分可独立验证**。最小验证实验:在 RoboCasa(已知子任务边界 ground truth)上,固定 [C1],只训 [C5] + L_HCS,验证 γ_sem_t 与 GT 边界 IoU > 0.5。0.5k traj,1 GPU,6 h。**写入循环效果仍需联立验证**。

---

#### [C5] 补充:γ 输出在 [C3]/[C4] 里如何起作用,以及"γ 真的有用吗"的诚实辩论

##### A. 一段话讲清楚

**[C5] 输出的 γ_geo_t 与 γ_sem_t 是两个标量乘子**,它们在 [C3] / [C4] 写头公式的**最外层**作为软门控,直接放缩本帧的写入幅度:

```
[C3]:  M_geo[v] += sg(γ_geo) · α · w · feature
                     ↑
                 标量乘子,作用在最外层

[C4]:  v_i     += sg(γ_sem) · w_i · φ(q_t) ⊗ v_t
                     ↑
                 同上
```

γ=1 全力写、γ=0 不写、γ=0.5 半力写——γ **不影响"写到哪个 voxel/slot"**(那由 [C3] token_to_voxel MLP 与 [C4] softmax-over-slots 决定),**只影响"写多狠"**。

**关键反直觉点**:与 [C3]/[C4] 的设计哲学不同,γ 的存在本身是**有争议的**——下面 §F 会展开"γ 是否真的必要"的诚实辩论。先讲清 γ 在公式里如何起作用,再讨论"它该不该存在"。

---

##### B. γ 在 [C3] 公式里的位置与数值效果

完整公式拆开看:

$$
\Delta M_{\text{geo}}^{(\ell)}[\text{neighbor}_j] \;\mathrel{+}=\; \underbrace{\mathrm{sg}(\gamma_{\text{geo}}^{(t)})}_{\text{[C5] 输出}} \;\cdot\; \alpha_\ell \;\cdot\; w_j \;\cdot\; \phi_\ell(h_t^{(i)})
$$

四项相乘的拆解:

| 项 | 数值范围 | 由谁决定 | 角色 |
|---|---|---|---|
| **sg(γ_geo)** | [0, 1] | **[C5] 动态输出** | 整帧统一的软门控 |
| α_ℓ | (0.5, 0.3, 0.2) | frozen 常数 | 三层分辨率间分配 |
| w_j | [0, 1],8 个权重和=1 | 三线性插值,几何决定 | 8 邻居 voxel 间分配 |
| φ_ℓ(h_t^{(i)}) | R^{64} 向量 | 可学投影 | 写入 feature 内容本身 |

**具体数值演示**(t=60,某个 voxel 邻居 j,w_j=0.4,fine 层 α=0.2):
```
γ_geo = 0.7(关键瞬间):
  ΔM_geo += 0.7 · 0.2 · 0.4 · feature  =  0.056 · feature

γ_geo = 0.1(平稳期,与上面同一 voxel):
  ΔM_geo += 0.1 · 0.2 · 0.4 · feature  =  0.008 · feature      ← 是上面的 1/7

γ_geo = 0(假设极端情形):
  ΔM_geo += 0  ← 本帧完全不写
```

**γ 整体放缩本帧所有 ~6000 次 += 操作的幅度**——256 token × 3 层 × 8 邻居,每次都被同一个 γ 乘。这是"音量旋钮"比喻的字面机制。

---

##### C. γ 在 [C4] 公式里的位置与数值效果

$$
v_i \;\mathrel{+}=\; \underbrace{\mathrm{sg}(\gamma_{\text{sem}}^{(t)})}_{\text{[C5] 输出}} \;\cdot\; w_t^{(i)} \;\cdot\; \phi(q_t) \otimes v_t
$$

| 项 | 数值范围 | 由谁决定 | 角色 |
|---|---|---|---|
| **sg(γ_sem)** | [0, 1] | **[C5] 动态输出** | 整帧统一的软门控 |
| w_t^{(i)} | [0, 1],64 个 softmax 和=1 | softmax(cos(q_t, k_i)) | 64 个 slot 间分配 |
| φ(q_t) ⊗ v_t | R^{256} 向量 | MLP_q、MLP_v 输出 | 写入内容 |

**具体数值演示**(t=60 红块入抽屉,假设 w_18=0.7):
```
γ_sem = 0.85(子任务边界):
  v_18 += 0.85 · 0.7 · 内容  =  0.595 · 内容    ← 强写入,slot #18 装上事件

γ_sem = 0.05(平稳期 t=45):
  v_18 += 0.05 · 0.7 · 内容  =  0.035 · 内容    ← 是上面的 1/17,slot 几乎不变
```

**γ 在两个写头里都是同一种角色**——乘在所有写入更新的最外层,统一放缩本帧的写入幅度。

---

##### D. 为什么 sg(γ) 必须乘在最外层 — stop-grad 的隔离作用

数学上 γ 也可以乘在中间层(混进 feature 计算里),但**乘在最外层 + sg** 配合起来有关键的工程意义——让 backward 梯度路径清晰:

```python
# [C3] 写入伪代码:
gamma_geo = ESPC(h_t, M_work)             # γ 来自 [C5] 的 forward
gamma_geo_sg = stop_gradient(gamma_geo)   # ★ 关键:sg 在写入瞬间

for token_i in range(256):
    feature = phi_l(h_t[i])
    for neighbor_j in range(8):
        M_geo[ell][neighbor_j] += gamma_geo_sg * alpha_l * w_j * feature
        #                         ↑
        #                       sg 在最外层
```

**sg(γ) 在最外层的效果**:
- L_main 经 [C8] read 反传到 M_geo,沿链式法则反传到 += 右侧
- 链式法则要求乘 ∂(sg(γ) · α · w · feature) / ∂γ
- 但 **sg(γ) 在 backward 里被 detach,梯度对 γ 求导 = 0**
- L_main 的梯度只能流向 α · w · feature 这一支(α frozen,所以只剩 feature → φ_ℓ → h_t)
- **L_main 完全不流入 [C5] ψ**

**这就是 HC-3 的工程化实现**:γ 必须在最外层 + sg,确保 L_main 训不动 [C5] 的 mixing 门控。如果 γ 不乘在外层(比如混进 feature 计算),sg 的阻断范围就模糊了——审查 Bug 4 当时的隐患就在此(虽然最终公式形态对了,但若代码实现把 γ 放进 feature 一侧,HC-3 隔离会形同虚设)。

---

##### E. γ ∈ [0, 1] 的设计意义

γ 是 sigmoid 输出,自然在 [0,1]——这不是技术限制,是有意选择:

- **γ ≤ 1**:[C5] 不能"过度强化"任何一帧——最多按 frozen 常数 α_ℓ 的全力写
- **γ ≥ 0**:[C5] 不能"减弱"已有 voxel/slot 内容——只能控制本帧写入,**不能写入负值去抵消旧内容**
- 这与 delta-rule 的"加性更新、不衰减"哲学一致——**γ 只控制本帧的写入,不能反向涂改 memory**

如果允许 γ ∈ [-1, 1](负值写入),会变成"软删除"机制——但那等价于引入了乘性衰减的另一种形式,撞 §3.2。

---

##### F. 诚实的设计辩论:**γ 真的有用吗?**

这是本架构里**最值得质疑的一个设计选择**——你的直觉合理:[C3]/[C4] 看起来"一直在记录",γ 这个系数似乎只是在调音量,**它真的解决了什么不靠它就解决不了的问题吗?**

让我把正反两面都摆出来,不偷懒。

###### F.1 反对 γ 必要性的论据(支持你的直觉)

**论据 1:[C3]/[C4] 已经有多重稀疏化机制,γ 的边际价值不明**

- [C3] 的 sparse scatter 已经天然只动 ~2000 voxel/帧,30000+ voxel 不动——大部分 grid 已经被"自动忽略"
- [C3] 的 token_to_voxel MLP 在 BC + L_PRH 监督下学会把背景 token 投到 grid 边界外或废弃 voxel——背景已经被自动过滤
- [C4] 的 softmax-over-slots 天然集中到 1-2 个 slot,其他 62 个 slot 几乎不变——非事件帧的 q_t 与所有 frozen 随机 key 的余弦相似度都低且接近,softmax 后权重弥散,**单个 slot 收到的 w 本来就小**
- **delta-rule 的 α 系数已经限制了每帧写入幅度**(0.2 for fine layer)

如果上述三层"自然稀疏化"已经能压住"无关帧不污染 memory",γ 这个额外的全局缩放因子的边际价值可能很小。

**论据 2:多帧累积是好事,γ 把它压低反而有害**

- [C3] 的核心好处之一是**同位置多帧 token 的加性累积 → emergent multi-frame fusion**(噪声抵消、信号强化)
- 假设机械手在同一位置 30 帧,大部分帧是"平稳运动"γ 低 → voxel 累积幅度被严重压低
- 不如 γ=1 保持每帧全力写——多帧累积自然得到强信号,无关帧的 token 已经被 token_to_voxel MLP 投到别处,不会污染

**论据 3:γ 引入了最大的赌注**

- γ 想 work 必须靠 L_HCS 训练 [C5](§F 选择 3 标 [弱],是整个架构最大的赌注)
- L_HCS 需要 [C10] HCS-H,而 E1 的 IoU @ 0.3 ≥ 0.4 是判决点(工程直觉概率 30-50%)
- 如果 E1 失败,γ 退化为 self-supervised prediction error 的 sigmoid——这本身就是 §3.1 提到的"代理信号永远不对齐"的直接体现
- **γ 的有用性高度依赖最不可靠的那个组件**

###### F.2 支持 γ 必要性的论据(架构层面的反驳)

**论据 1:γ 解决的是"长程平稳期信息淹没"问题**

- 假设无 γ,平稳搬运期间机械手在 (0.3, 0.3, 0.4) 持续 30 帧,fine voxel (9,9,12) 累积幅度可达 ~3.0(0.2 × 0.4 × 30 ≈ 2.4)
- 而真正关键的"红块入抽屉"瞬间只有 1-3 帧的写入,累积幅度仅 ~0.3
- read 时 cross-attn 的 query 会被"机械手所在 voxel 的强信号"主导,**关键瞬间被噪声淹没**
- γ 的作用是**把平稳期写入幅度压低 5-10 倍**,让关键瞬间相对突出——不是绝对幅度,是相对对比

**论据 2:γ 提供"任务相关性"的额外监督路径**

- token_to_voxel MLP 学的是"什么 hidden state 投到哪个 3D 位置"——它**学不到"这一帧整体是不是关键"**(那是帧级语义,不是 token 级几何)
- γ_geo / γ_sem 是**帧级**信号——整帧统一缩放,补充 token-level 路由所欠缺的"帧级显著性"
- 没 γ 的话,无论关键帧还是平稳帧都按同样 α 写——丢失了"帧级显著性"这一维度

**论据 3:γ 是 [C10] HCS-H 唯一的 attach point**

- 即使 γ 本身价值边际,**[C10] HCS-H 必须作用到某个组件上**——否则 hindsight saliency 信号无处安放
- 如果删 γ,L_HCS 找不到归属,整个"hindsight 监督"路线断掉
- §F 选择 3 标 [弱] 正是承认了这一点——但即便如此,**保留 γ 是为整个 hindsight 路线留下接口**,不是 γ 本身的价值

###### F.3 一个直接的实验判决

**Ablation 设计**:在 MVP 配置下,对照三种 γ 策略,看长程 SR 差异:

| 配置 | γ 来源 | 预期长程 SR(>150 步) |
|---|---|---|
| **γ_const = 1.0**(全力写,无门控) | 常数 1 | 基线,中等(可能被平稳期信息淹没) |
| **γ = sigmoid(prediction error)** | self-supervised(MVP fallback) | 微弱提升 / 持平(代理信号不准) |
| **γ from L_HCS**(full 版本) | hindsight 监督 | 显著提升(若 E1 通过) |

**这个 ablation 是 §F.6 表里没列的关键一项**——如果"γ_const = 1.0"与"γ from L_HCS"在长程 SR 上差异 < 5%,**γ 这个设计要被推翻**,降级为常数。我会把这一项补到 §F.6。

###### F.4 我的诚实回答

你的直觉**对了一半**——γ 的边际价值确实模糊,在没有 L_HCS 时它可能是冗余的。但它的作用不是"调音量"那么简单:它是**帧级显著性信号的载体**(token 级路由不能替代),也是**整个 hindsight 路线的 attach point**(没 γ 整个 [C10][C5] 链路没有作用对象)。

**MVP 阶段的处理**:γ 退化为 self-supervised prediction error 的 sigmoid——这版本的 γ 边际价值最弱,你的质疑对它最适用。**如果 ablation 显示 γ_const=1 与 self-supervised γ 持平,MVP 就直接用 γ=1 的简化版,删 [C5]**——这是合理的设计简化路径,文档里应当显式列出。

**Full 版本的依据**:γ 的真正价值绑定在 L_HCS 上——如果 E1 通过 + ablation 显示 γ from L_HCS 显著优于 γ_const=1,那 γ + [C5] + [C10] 这一整套才能站住脚。否则就是过度工程。

---

##### G. 一句话总结

γ_geo 与 γ_sem 是 **[C5] 输出的两个标量乘子,乘在 [C3]/[C4] 写入公式的最外层**,直接放缩本帧整体写入幅度——γ=1 全力写、γ=0 不写。乘在最外层 + sg(γ) 配合起来确保 L_main 不流入 [C5](HC-3 工程化实现)。**γ 自身价值有争议**——三层自然稀疏化(sparse scatter、token_to_voxel 学忽略背景、softmax 集中)已经过滤了大部分无关帧;γ 的边际价值绑定在 L_HCS 与"帧级显著性"上。**MVP 阶段建议加一个 ablation:γ_const=1 vs self-supervised γ vs L_HCS γ,如果前两者持平就直接简化为 γ=1**——这条 fallback 应当列入 §F.6 表。

---

### [C6] 几何记忆 grid M_geo

**前置依赖**:[C3] 写入。

**解决约束**:HC-1(几何容量充裕,具体数字见 §E)、HC-7、§3.2(无门控乘子,代数上不衰减,**仅在 grid 未满时**——满后 LRU 丢弃 stale 条目,不衰减但会丢失,见 §H Trade-off 5)、§3.4。

**接口契约**:
- 输入:[C3] 写入增量
- 输出:M_geo ∈ R^{(8³+16³+32³)×d_g},d_g=64
- 读端:[C8] 经 N_geo_q=16 个 learnable position queries 三线性采样

**内部 operator**:三层 voxel grid(总 ~4.2 万元)。

**最近原型**:NICE-SLAM。差别:无 RGB-D 监督,任务驱动。

**可学性**:无参数。

**梯度路径**:L_main, L_PRH 经 [C3]。

**可独立验证性**:见 [C3]。

---

### [C7] 语义 slot bank M_sem

**前置依赖**:[C4] 写入、[C12] CSM 提供 slot 重要度排序(用于 LRU 丢弃)。

**解决约束**:HC-1、HC-7、§3.2(未满时)、§3.4。

**接口契约**:
- 数据张量:M_sem.v ∈ R^{B×K_s×d_s},M_sem.k ∈ R^{B×K_s×d_s},K_s=64,d_s=256 [标定值]
- **外部状态(episode 内布尔 mask)**:`slot_free ∈ {0,1}^{B×K_s}`,1=空槽可用,0=已占用
- Episode 间硬清空:M_sem.v ← 0、slot_free ← 1、M_sem.k 重新随机(放弃跨 episode 学习,见 §H Trade-off 2)

**内部 operator**:K_s 个 slot,key k_i 在 episode 开始时随机初始化 + frozen,value v_i 通过 [C4] delta-rule 累加。容量满时 LRU 丢弃(见 [C4] 步骤 5)。**丢弃语义(D5 修订)**:`v_i ← 0`、`slot_free[i] ← 1`、`k_i 不变`——保留 frozen key 让未来事件可重用同一物理 slot 的路由空间。

**slot_free mask 的两处用法**(写入与读出都必须用):
1. **写入路由(在 [C4] softmax 前)**:logit_i ← logit_i − 1e9 · slot_free[i] 之后再 softmax——把空槽变为 0 概率(不被 delta-rule 累加污染),避免数值未定义的 cos(q, v_i=0)。
2. **读出端(在 [C8] cross-attn key 上)**:同样 logit penalty,空槽不参与 attention。

**最近原型**:XMem long-term pool + Infini-attention delta-rule。差别:用 LRU 丢弃替代 prototype 平均,以保 delta-rule 代数性质。

**可学性**:无新参数。

**梯度路径**:L_main 经 v_i 到 [C4] MLP_v。L_CSM 经 leave-one-out 到 [C4]。

**可独立验证性**:见 [C4]。

---

### [C8] 读出接口(Read Interface)

**前置依赖**:[C2][C6][C7]。

**解决约束**:HC-2(全可微 read)、HC-6(实测延迟见 §H Trade-off 3)。

**接口契约**:
- 输入:M_work, M_geo, M_sem, h_t
- 输出:c_t ∈ R^{(N_q + K_w)×d_h}
- 频率:每步

**内部 operator**:
1. N_q=16 learnable query 做 cross-attn 到 M_work + M_sem(token + slot 混合 KV,KV 总数 = K_w·N + K_s = 8·256 + 64 ≈ 2k)
2. N_geo_q=16 spatial query 三线性采样 M_geo 的 coarse 层为主(8³=512 voxel),mid/fine 层只在 coarse 命中后局部精化(避免遍历 5e4 voxel)
3. 拼接、LayerNorm、加位置编码

**最近原型**:Memorizing Transformer + Perceiver IO。

**可学性 / 参数量**:cross-attn ~5M,query embeddings ~50K。

**梯度路径**:L_main 经 cross-attn 反传。**关键**:对 L_PRH 路径上的 query 投影矩阵 sg,只让 KV 路径反传——避免 L_PRH 经 query → h_t 污染 [C1]。

**可独立验证性**:不必。

---

### [C9] Action Expert

**前置依赖**:[C8] c_t、[C1] h_t (CLS)。

**解决约束**:无新增。

**接口契约**:输入 c_t, h_t (CLS);输出 a_t ∈ R^7。

**内部 operator**:Flow matching head(类似 π0)+ LoRA r=16,主体 frozen。

**最近原型**:π0(Black et al. 2024)。差别:无。

**可学性**:LoRA ~3M。

**梯度路径**:L_main → adapter。L_CSM 经 frozen [C9](自然 sg)。

**可独立验证性**:不必。

---

### [C10] Hindsight Causal Saliency Head(HCS-H,training-only)

##### 一段话先讲清楚

**[C10] 在做什么**:在离线 demonstration 上,**用"事后已知的未来动作 a*_{t+Δ}"反推"第 t 帧值不值得记"**——为每帧产出一个 saliency 真值 γ̂_geo_t / γ̂_sem_t,然后通过 L_HCS 训 [C5] ESPC 把它的 γ_*_t 推向 γ̂_*_t。

**这是 §3.1"写入时无法预知未来"这一因果不可能问题的唯一突破口**——在线时不知道未来,但**离线 demonstration 上未来是已知的**,所以可以反推真值再 baked 进 [C5] 的 prediction-error 校准里。**部署时 [C10] 完全消失**,影响沉淀在 [C5] 参数里。

**它是整个架构最大的赌注**(§F 选择 3 标 [弱]):核心未知是"Jacobian saliency 在真实 expert demo 上的信噪比是否足够"。E1 是判决点,失败则 fallback 到 MVP 简化版。

---

##### 接口契约 + 前置依赖

| | |
|---|---|
| 输入(offline) | 整段连续 trajectory τ = (o_{1:T}, a*_{1:T}) |
| 输出 | γ̂_geo_t, γ̂_sem_t ∈ [0,1],经 sg 后作为 L_HCS 的 BCE target |
| 频率 | 训练时每 mini-batch 一次,**部署时不跑** |
| 数据要求 | 轨迹必须未被 chunk 截断,T ≥ max(Δ)+1 |
| 已验证可用 dataset | BridgeV2 raw HDF5、RoboCasa sim;**RMBench/MemoryBench 不公开**(详见 §I.0) |

---

##### 内部工作流(三步合成 γ̂)

**Step 1 — Jacobian saliency(主信号)**

对未来 Δ ∈ {4, 16}(显存允许时加 64),通过 frozen base policy 反向计算:

$$
J_t^{(\Delta)} = \left\|\frac{\partial a^*_{t+\Delta}}{\partial o_t}\right\|_F
$$

**直觉解读**:"未来 Δ 步的真实最优动作 a*_{t+Δ} 对当前帧观测 o_t 有多敏感"——敏感度高 = 当前帧信息对未来决策很关键 = 这帧值得记。

**Step 2 — RUDDER 二阶信号(辅助)**

训练一个轻量 LSTM g_θ 回归整 episode 的成功率 R̂,逐帧 saliency 取一阶差分:

$$
c_t = g_\theta(\tau_{\le t}) - g_\theta(\tau_{\le t-1})
$$

**直觉解读**:"看完前 t 帧后,episode 成功概率比看完前 t-1 帧后变化了多少"——变化大 = 这帧承载了显著进度推进。

这是 RL 文献(Arjona-Medina 2019)的 return redistribution 技术,把 episode 级稀疏 reward 转换为逐帧 saliency。

**Step 3 — 通道分离 + 归一化**

frozen patch-level grad-cam 把 Jacobian 分解为局部空间贡献 J_geo 与全局贡献 J_sem;两个通道分别融合 RUDDER 信号 c_t,整段轨迹 z-score 后过 sigmoid:

$$
\hat\gamma_{\text{geo},t} = \sigma\!\left(\text{z-score}(J_\text{geo,t} + \beta \cdot c_t)\right) \quad ; \quad \hat\gamma_{\text{sem},t} = \sigma\!\left(\text{z-score}(J_\text{sem,t} + \beta \cdot c_t)\right)
$$

输出送给 L_HCS 作为 BCE target(L_HCS 完整定义见 §D L2)。

---

##### 关键子决策:Base policy 选什么

[C10] 算 Jacobian 需要一个 frozen base policy 当"标尺"。**这个选择影响 J 的语义,不能含糊**:

| 选项 | 优点 | 缺点 | 何时用 |
|---|---|---|---|
| **π0.5**(无记忆预训练) | 立即可用,无 chicken-and-egg | J 反映"无记忆策略对当前帧依赖",是任务相关帧的 lower bound,信号偏弱 | **E1 第一轮(默认起点)** |
| **CHIME early checkpoint** | J 信号更锐,反映长程信息需求 | chicken-and-egg:早期 CHIME 还没学会用记忆,J 有偏 | E1 通过 + E2 训完后切换 |
| π0.5 fine-tuned on target | 中间方案,信号较准 | 工作量加倍,需对每个数据集重训 | 资源充裕时备选 |

---

##### 解决哪些约束

| 约束 | [C10] 怎么对应 |
|---|---|
| **HC-2 梯度可达** | hindsight saliency 直接给旧帧梯度标签,绕开跨百步 BPTT |
| **HC-4 aux loss 必要** | L_HCS 是给 [C5] 写入门的独立 aux loss,与 L_main 通过 sg 隔离 |
| **§3.1 因果不可能** | 承认在线无法预知未来,但用**离线已知未来**反推训练目标——唯一突破口 |
| **§3.3 跨步信用(部分)** | 每帧产生 saliency label,梯度无需跨百步 BPTT;但 Δ 受显存上限制约(§H Trade-off 7) |
| **HC-5(部分)** | γ̂ 是连续标量替代 0/1 分类器;但 [C10] 自身仍是单组件,内部 Jacobian + RUDDER 是 fork 不是独立校验(§H Trade-off 6) |

---

##### 梯度与参数

- **可学参数**:仅 RUDDER LSTM g_θ ~8M。Jacobian 部分用 frozen base policy,grad-cam 也是 frozen,无可学参数。
- **梯度流向**:L_HCS → [C5] ψ + projections。**γ̂ 在 BCE target 侧 sg,不会反向训 [C10] 自己**。L_HCS **不流入** [C1][C3][C4][C6][C7][C9](与 §D L2 stop-grad 列表一致)。

---

##### 计算与显存预算(Δ 选择的硬约束)

| 项 | 估算(基于 N=256 token, d_h=1152, batch=32) |
|---|---|
| Jacobian 单帧 | J ∈ R^{7×N×d_h} ≈ 8 MB |
| Jacobian 全 batch × 3 个 Δ | ≈ 750 MB |
| frozen base policy 跨 64 步 activation | ≈ 30 GB at fp16(需 gradient checkpointing) |
| 单 step backward 计算开销 | ~3× 主路径 backward,贡献总训练时间 ~40% |
| **OOM 风险** | ~30%(在 80GB H100 / A800 上) |
| **Fall-back** | Δ 默认 {4, 16} 不含 64;若仍 OOM,batch=16 |

详细 epoch 预算见 §H Trade-off 3。

---

##### 最近原型

**RUDDER**(Arjona-Medina 2019,return redistribution)+ **PSM-CWM HMD**(Hindsight Manipulation Distillation)。
**差别**:RUDDER 原本用于 RL value redistribution → 这里改作 saliency target;PSM-CWM 只用单一 Jacobian → 这里加 RUDDER 作二阶信号补充。

---

##### 可独立验证性 + E1 判决点(关键)

**E1**(workflow 第一道判决):在 ~1k BridgeV2 trajectories 上独立运行 [C10],计算 J_t^{(16)} 的归一化结果,检查与人工事件标注的 IoU @ 0.3。

| 结果 | 处理路径 |
|---|---|
| **IoU ≥ 0.4 通过** | 继续走 full 版本,E2/E3 切换到 CHIME early checkpoint base policy |
| **IoU < 0.4 失败** | **disable [C10][C12][C13] 与 L_HCS、L_CSM 全套**,fallback 到 §F 选择 3 简化版(γ 退化为纯 prediction-error self-supervised + L_PRH)。简化版 publishable claim 退化为"打 HC-3 + §3.2 + §3.4 三条,放弃 §3.1 后视回应"。**继续 E2-E5,不阻塞工作流** |

**注意**:BridgeV2 没有现成的 frame-level event 标注,E1 需要 1-2 人周手标 100-200 traj(详见 §I.0)——这是文档原草案漏掉、审查找补的隐藏工作量。

---

### [C11] Predictive Read Head(PRH,training-only)

##### 一段话先讲清楚

**[C11] 在做什么**:训练时把当前 memory 的读出 m_t 喂给三个独立 MLP head,**预测未来 k ∈ {4, 16, 64} 步的观测 o_{t+k} 与最优动作 a*_{t+k}**。预测准 = memory 装的内容"对未来有用";预测不准 → 梯度反传修正写头(逼写头把"对未来有用的"信息更多地写入 memory)。

**核心思想**:§3.1 承认无法预知"未来需要什么",但**要求 memory 内容对未来可预测**——把"被需要"目标转换成"可预测"代理。这是 MERLIN(Wayne et al. 2018)思想搬到 VLA 上。

**与 [C10] 的核心差异**:[C10] 监督的是"何时记"(γ saliency target),[C11] 监督的是"记的内容是否对未来有用"。两者是**正交的两条 aux loss**——共同回应 HC-4。**风险比 [C10] 低很多**:MERLIN 是成熟先例,且不依赖 hindsight Jacobian 这种没保证的信号。

---

##### 接口契约 + 前置依赖

| | |
|---|---|
| 前置依赖 | [C8] m_t 读出、整段 trajectory(需要未来 k 步的 o_{t+k} / a*_{t+k} 作监督) |
| 输入(训练时) | m_t,经 [C8] 但 query 投影矩阵在 L_PRH 路径上 sg |
| 输出 | 6 个预测向量(3 个 horizon × {obs, action}) |
| 频率 | 训练时每 mini-batch 每帧 |
| 部署时 | **完全不跑** |

---

##### 工作流:三 horizon 多任务监督

```
训练时,每帧 t:
  m_t = [C8].read(M_work, M_geo, M_sem)
  
  # 三个独立 prediction MLP × 2(observation + action)
  ô_{t+4}  = MLP_o_4(m_t);   â*_{t+4}  = MLP_a_4(m_t)
  ô_{t+16} = MLP_o_16(m_t);  â*_{t+16} = MLP_a_16(m_t)
  ô_{t+64} = MLP_o_64(m_t);  â*_{t+64} = MLP_a_64(m_t)
  
  L_PRH = Σ_{k∈{4,16,64}} [‖ô_{t+k} - o_{t+k}‖² + α_a · ‖â*_{t+k} - a*_{t+k}‖²]
```

α_a = 1.0 [标定值]。详细 loss 公式见 §D L3。

---

##### 为什么三个 horizon 而不是单个

不同 k 监督不同时间尺度上的 memory 内容:

| Horizon | 主要监督 | 主要受影响组件 |
|---|---|---|
| **k=4** | 短期细节是否被记录 | [C2] FIFO + [C3] geo 写头近期写入 |
| **k=16** | 中期事件序是否被记录 | [C7] M_sem slot bank 的近期事件 |
| **k=64** | 长程 belief state 是否被记录 | [C6] M_geo 跨遮挡 + [C7] 长期 slot |

单一 horizon 的失败模式:只用 k=4 → memory 只学短期细节,长程没监督;只用 k=64 → 短期细节没压力。**多 horizon 同时施加,memory 必须对各时间尺度同时可预测**。

---

##### 数值演示(抽屉任务 t=60)

t=60 红块刚入抽屉:
- m_60 含 slot #18(红块入抽屉)+ voxel(抽屉内位置)
- L_PRH 要求 m_60 能预测 o_{76}(关抽屉中)、a*_{76}(关抽屉动作)
- 如果 [C4] 路由正确(slot #18 真装上了"红块入抽屉")→ m_60 含足够信息 → 预测准 → L_PRH 低
- 如果 slot 路由错(#18 没装上) → m_60 漏关键信息 → 预测不准 → 梯度反传修正 [C4] MLP_q/MLP_v

**L_PRH 是写头的"完成度考核"**——不直接说"写什么",而是说"你写的应该让 m_t 能预测未来"。

---

##### 解决哪些约束

| 约束 | [C11] 怎么对应 |
|---|---|
| **HC-4 aux loss 必要** | L_PRH 是给写头 + memory 内容的独立监督(与 L_main、L_HCS、L_CSM 互相 sg 隔离) |
| **§3.1 因果不可能(部分)** | 承认无法直接知道"未来需要什么",但用"未来可预测"作代理——比 [C10] 的"事后真值"弱但更稳健 |

---

##### 梯度与参数

- **可学参数**:6 个独立 MLP head × ~1.5M = ~9M。
- **梯度流向**:L_PRH → MLP head → m_t → cross-attn KV 路径 → [C3][C4] 写头 + [C6][C7] memory 内容。
- **关键 sg**:[C8] 的 query 投影矩阵在 L_PRH 路径上 sg——**避免 L_PRH 经 query → h_t 污染 [C1] LoRA**(审查 Bug 9 的修复)。

---

##### 最近原型 + 差别

**MERLIN**(Wayne et al. 2018, DeepMind)的 read-reconstruct loss:从 memory 读出后预测**当前**观测重建。
**差别**:PRH 预测**未来 k 步**而非当前——这把"memory 应可重建过去"改为"memory 应可预测未来",更直接对齐 §3.1 的"未来需求"代理。

---

##### 可独立验证性

**完全可独立验证**——固定 [C1][C9],只训 [C3][C4][C6][C7] memory + [C11] PRH(**无 L_main**),看 m_t 是否真的能预测 o_{t+16}。

| 指标 | 通过门槛 |
|---|---|
| k=16 时 MSE | 显著低于 baseline `m_t = h_t alone`(只用当前帧、不用 memory) |
| 数据需求 | 1k traj |
| 资源 | 1×H100,8 小时 |

如果连这个 baseline 都打不过,说明 memory 写入完全无效——这是个早期 sanity check,**适合 E3 单独跑**(详见 §I)。

---

### [C12] Counterfactual Slot Mask(CSM,training-only)

##### 一段话先讲清楚

**[C12] 在做什么**:训练时随机抽 [C7] M_sem 的几个 slot,用 leave-one-out(把那个 slot value 置零)看 [C9] action expert 输出分布的变化幅度——**变化大 = 那个 slot 对决策有因果贡献**。L_CSM 用这些"因果重要度" w_i 训 [C4] MLP_q/MLP_v,**鼓励 slot 之间的重要度差异化**(避免所有 slot 装相似中等内容)。

**核心思想**:slot bank 的最大风险是 **slot collapse**——所有事件被路由到同一 slot,或多个 slot 装相似内容,失去区分性。[C12] 用反事实因果度量强制 slot 异质化:**每个 slot 必须装"删掉它会让决策显著变化"的内容**。

**双重作用**:除了作为 aux loss 反传到 [C4] 写头,[C12] 算的 w_i 还被 **[C7] LRU 丢弃决策复用**——容量满时丢"因果重要度最低"的 slot,而非 last-read time 最旧的。

---

##### 接口契约 + 前置依赖

| | |
|---|---|
| 前置依赖 | [C7] M_sem、frozen [C9] action expert |
| 输入(训练时) | m_t、随机选的 4 个 slot 索引 |
| 输出 | 因果重要度 w_i = KL(π(a\|m_t) ‖ π(a\|m_t \ slot_i)) |
| 频率 | 训练时每 mini-batch 抽 4 slot 算 |
| 部署时 | **完全不跑**;但 w_i 的训练时累积统计被 [C7] LRU 复用 |

---

##### 工作流:每 mini-batch 抽 4 slot 做 leave-one-out

```
训练时,每个 mini-batch:
  1. 随机选 4 个 slot 索引 i ∈ {1..64}
  
  2. 对每个被选 slot i:
     m_t       = [C8].read(M_work, M_geo, M_sem)            ← 完整读出
     m_t^{-i}  = [C8].read(M_work, M_geo, M_sem|v_i=0)      ← 第 i 个 slot value 置零
     
     a_dist_full     = frozen[C9](m_t)
     a_dist_minus_i  = frozen[C9](m_t^{-i})
     
     w_i = KL(a_dist_full || a_dist_minus_i)                ← slot i 的因果重要度
  
  3. L_CSM = -Var_i(w_i) - β · log(Mean_i(w_i))   β=0.1 [标定值]
```

**关键设计**:
- **frozen [C9]** 保证 L_CSM 不会反向训 [C9](自然 sg)
- 只抽 4 slot 而非全 64(全做计算太贵,4 个采样估计够)
- 每 mini-batch 抽不同 4 slot,长期覆盖所有 slot

---

##### Loss 两项各起什么作用

```
L_CSM = -Var_i(w_i) - β · log(Mean_i(w_i))
        ↑                    ↑
       项 A(异质化)         项 B(利用率)
```

| 项 | 作用 | 防止的失败模式 |
|---|---|---|
| **−Var_i(w_i)** | 鼓励 slot 重要度方差大——一些 slot 高、一些低 | 防止"所有 slot 装相似中等内容"的退化 |
| **−β·log Mean_i(w_i)** | 鼓励整体 slot 重要度高 | 防止"只有 1-2 个 slot 真有用,其他 62 个全是垃圾" |

两项加起来:slot 应当差异化但整体都有用——异质化与利用率的平衡。

---

##### 数值演示(抽屉任务 t=90)

t=90 决策"是否重开抽屉":m_t 含 slot #18(红块入抽屉)+ slot #5(抽屉曾被开)+ slot #27(关闭进行中)。

**Leave-one-out slot #18**:
- m_t^{-18} 不含"红块入抽屉" → action expert 不知道红块在内 → 决策可能变成"重开看一眼"
- KL(full ‖ minus_18) **大** → w_18 高

**Leave-one-out slot #5**:
- m_t^{-5} 不含"抽屉曾被开"→ 但 t=90 抽屉马上要关,知不知道"被开过"对当前动作影响小
- KL **小** → w_5 低

**Var(w_i) 大** → L_CSM 项 A 满足。这告诉 [C4] MLP_q/MLP_v:把"红块入抽屉"分到独立 slot #18 是对的(那 slot 真承担了因果作用)。

---

##### 解决哪些约束

| 约束 | [C12] 怎么对应 |
|---|---|
| **HC-4 aux loss 必要** | L_CSM 是给 [C4] 写头的独立监督,与 L_main、L_HCS、L_PRH 互相 sg 隔离 |
| **§3.4 异质共表示有损** | 强制 slot 间结构性差异——每个 slot 必须承担可识别的因果角色,不能全装相似内容 |

辅助作用:为 [C7] LRU 丢弃提供 slot 重要度排序——容量满时丢 w_i 最低的(而非 last-read time 最旧的)。

---

##### 梯度与参数

- **可学参数**:0(用 frozen [C9])。
- **梯度流向**:L_CSM → [C4] MLP_q/MLP_v(经 softmax 重新分配 + delta-rule 累加)。
- **关键 sg**:经 frozen [C9] 自然 sg——L_CSM **不会**反传修改 [C9],否则 [C9] 会被训成"对 slot 缺失敏感"以 hack 这个 loss。

---

##### 计算成本(为什么抽 4 slot 而非全 64)

每个 leave-one-out 需要:
- 1 次 frozen [C9] forward(完整 m_t)
- 1 次 frozen [C9] forward(m_t^{-i})

**抽 4 slot × batch 32**:
- 每 mini-batch 增量 = 4 × 2 × 32 = 256 次 frozen [C9] forward
- 占总 forward 计算 ~30-40%

**可调**:抽样数 4 → 2 时成本减半,但 slot 间方差估计更糙。MVP 默认 4。

---

##### 最近原型 + 差别

**Mesnard et al. 2021**(Counterfactual Credit Assignment, ICML)的 future-conditioned counterfactual baseline:`A_t^CF = Q − V(·|F)` 用于 RL advantage estimation。
**差别**:Mesnard 用于 RL 优势估计,这里改作 slot 因果重要度信号——把"哪个 action 因果贡献大"换成"哪个 slot 因果贡献大"。

---

##### 可独立验证性

**必须与 [C7] 联立验证**——单独 [C12] 没意义。

| 验证方式 | 通过判据 |
|---|---|
| 开 / 关 [C12] 看 slot 利用率分布 | 关 L_CSM 时:slot 分配可能退化为 hash collision(80%+ 写入集中到一个 slot) |
| | 开 L_CSM 时:slot 分配应分散到 5-10 个 slot |

---

### [C11] vs [C12] 一表对比

| | [C11] PRH | [C12] CSM |
|---|---|---|
| 监督什么 | memory 内容**预测未来**的能力 | slot **彼此因果异质**的程度 |
| 防什么失败 | memory 装垃圾 / 装"对未来无用"内容 | slot collapse(所有 slot 装相似内容) |
| 作用对象 | [C3][C4] 写头 + [C6][C7] memory 全部 | **仅 [C4] MLP_q/MLP_v**,影响 [C7] slot 路由 |
| 计算开销 | 6 个 prediction head forward(轻) | 抽 4 slot × 2 frozen [C9] forward(中) |
| 风险 | **低**(MERLIN 成熟先例) | 中(Mesnard 在 RL 上验证,迁移到 saliency 是 adaptation) |
| MVP 取舍 | **保留**(轻量、低风险) | **可砍**(纯 ablation 价值,MVP 不必;E1 失败时砍掉) |

[C11] 是普适的"memory 质量考核",[C12] 是专门针对 slot bank 的"异质化强制"。两者与 [C10] 一起构成训练时的三个独立 aux 监督——这就是 HC-4 aux loss 必要性的工程化实现。

---

### [C13] Reverse Hindsight Auxiliary(SWR-style,training-only)

**前置依赖**:[C10]。

**解决约束**:§3.3 部分(给 ESPC 反向因果信号)。

**接口契约**:
- 输入:trajectory replay buffer
- 输出:在反向时间方向上算 ∂a*_{t-Δ}/∂o_t 作为额外 BCE 标签喂给 [C5]

**内部 operator**(已修正原草案 Bug 8——不再做 forward 反向):
1. 在原始 forward trajectory 上,[C10] 已算出 J^{(forward,Δ)}_t = ∂a*_{t+Δ}/∂o_t。
2. **反向版本**:[C10] 同时算 J^{(reverse,Δ)}_t = ∂a*_{t-Δ}/∂o_t。这只是 Jacobian 的另一个方向,不需要把网络反向 forward。
3. 当某些任务存在反向因果(罕见,但例如柔性物体接触反应时),反向 J 会显著非零。把它合入 γ̂_*_t = sigmoid(α·J_forward + (1-α)·J_reverse),α=0.8 [标定]。

**最近原型**:Foster 2017 的 sharp-wave ripple 反向重放(神经科学)。差别:神经科学是 forward 反向跑网络;ML 这里只是 Jacobian 双向计算。**[纯新设计 in ML 文献],我承认这是 v2 第二个最弱依据组件。**

**可独立验证性**:无独立指标。Ablation:开 / 关 [C13],看长程 SR 差异。

---

## §D 训练目标的完整形式

设训练样本 trajectory τ = (o_{1:T}, a*_{1:T})。

### L1: 主任务 BC loss(Flow Matching)

$$\mathcal{L}_{\text{main}} = \mathbb{E}_{t, \epsilon \sim \mathcal{N}(0,I)} \left[ \| v_\theta(a^*_t + \sigma\epsilon, t, c_t) - \epsilon \|_2^2 \right]$$

- 监督:demonstration a*_t。
- 约束:基础 BC,无直接 HC/§。
- 影响参数:[C1] LoRA, [C8] cross-attn, [C9] LoRA, [C3][C4] 写头(经 sg(γ),不流入 [C5])。
- Stop-grad:**sg(γ_geo)、sg(γ_sem)在 [C3][C4] 写入**,阻断 L_main → [C5]。
- 可消融:基础 loss,不可去。

### L2: Hindsight Causal Saliency loss(L_HCS)

$$\mathcal{L}_{\text{HCS}} = \sum_t \sum_{* \in \{\text{geo, sem}\}} \text{BCE}\big(\gamma_*^{(t)}, \mathrm{sg}(\hat\gamma_*^{(t)})\big)$$

- 监督:[C10] 离线 saliency。
- 约束:HC-2、HC-4、HC-5(部分)、§3.1、§3.3(部分)。
- 影响参数:[C5] ESPC(ψ + projections)。
- Stop-grad:**sg(γ̂)在 BCE 目标侧**;**sg 切断 L_HCS → [C1][C3][C4][C6][C7][C9]**。
- 可消融:**立即可见**。删掉,γ_*_t 退化为 self-supervised prediction error;长程 SR 显著下降(预期)。

### L3: Predictive Read loss(L_PRH)

$$\mathcal{L}_{\text{PRH}} = \sum_{k \in \{4,16,64\}} \mathbb{E}_t \left[ \| \hat o_{t+k} - o_{t+k} \|_2^2 + \alpha_a \| \hat a^*_{t+k} - a^*_{t+k} \|_2^2 \right]$$

α_a = 1.0 [标定值]。

- 监督:trajectory 自身未来。
- 约束:HC-4、§3.1。
- 影响参数:[C3][C4] 写头,[C6][C7] memory,[C11] PRH MLP。
- Stop-grad:**sg(query 投影矩阵)在 [C8] 阻断 L_PRH → [C1]**。允许 L_PRH 与 L_main 共享 [C3][C4]。
- 可消融:**长程才显现**。短程(<50 步)无差;长程(>100)预期显著下降。

### L4: Counterfactual Slot Mask loss(L_CSM)

$$\mathcal{L}_{\text{CSM}} = -\text{Var}_i (w_i^{(t)}) - \beta \cdot \log\text{Mean}_i(w_i^{(t)})$$

(注:符号修正——最大化方差 + 平均的 log,所以 loss 取负;β=0.1 [标定值]。)

- 监督:m_t 上 leave-one-slot-out + frozen [C9],无外部标注。
- 约束:HC-4、§3.4。
- 影响参数:**仅 [C4] MLP_q/MLP_v**。
- Stop-grad:**sg 经 frozen [C9]**(自然)。
- 可消融:中长程才显现。删掉,slot 分配可能退化为 hash collision。

### L5: Geometric Consistency loss(L_GC,可选,**MVP 版默认关闭**)

$$\mathcal{L}_{\text{GC}} = \mathbb{E}_t \left[ \| \text{render}(M_{\text{geo}}^{(t)}, \text{view}) - o_t^{\text{view}} \|_2^2 \right]$$

仅在 RGB-D / 多视角数据下启用。MVP 阶段 disable。

### L_aux: Attention Entropy 正则(MVP 默认开启,小系数)

$$\mathcal{L}_{\text{aux}} = -\lambda_{\text{ent}} \cdot \mathbb{E}_t \left[ H(\text{attn weights of [C8] over M\_work}) \right]$$

防止 [C8] 把 [C2] FIFO 训成"只读最新 1-2 帧"的短视读出(Bug 7 对策)。λ_ent=0.01 [标定值]。

### 总损失

$$\mathcal{L} = \mathcal{L}_{\text{main}} + \lambda_1(\text{step}) \cdot \mathcal{L}_{\text{HCS}} + \lambda_2 \mathcal{L}_{\text{PRH}} + \lambda_3 \mathcal{L}_{\text{CSM}} + \lambda_4 \mathcal{L}_{\text{GC}} + \mathcal{L}_{\text{aux}}$$

**λ_1 是 schedule(隐式课程,见 proposal §7)**:
- step < `step_E1_pass`(M1 出口): λ_1 = 0(L_HCS 完全不接通,ψ 仅靠 self-supervised prediction error 训)
- `step_E1_pass` ≤ step < `step_E1_pass + 5000`: λ_1 线性 anneal 0 → 0.3
- step ≥ `step_E1_pass + 5000`: λ_1 = 0.3 [恒定标定值]

λ_2=0.5, λ_3=0.1, λ_4=0.2 [全部标定值,静态]。

**为什么 λ_1 必须 schedule**(不能从第一步就 0.3):cold-start 时 ψ 输出还在初始化噪声里,γ̂ 也未必准——立即接通 L_HCS 会把 ψ 训成 BCE 拟合噪声目标。proposal §7 line 313-325 论证这是隐式课程的关键一环;实证上从 0 anneal 到 0.3 vs 静态 0.3 的差异是 §I.4 red flag #2 的对策之一。

**Mini-batch 切片策略**:严格按 episode 切,不允许跨 boundary 的 (t, t+k) pair 进 L_PRH 或 L_HCS;[C13] 反向 Jacobian 也只在单个 episode 内反向。

---

## §E HC/§ ↔ 组件 ↔ Loss 三向映射表

| 约束 | 主要回应组件 | 主要回应 Loss | 状态 | 备注 |
|------|------|------|------|------|
| HC-1 容量 | [C2][C6][C7] | (结构) | **已回应**(结构充裕,非真正瓶颈) | 总容量 ~12.9 MB at fp32 / 6.5 MB at fp16,远超 H_required ~100 bits。HC-1 在所有现代 VLA 上都不是瓶颈(MemoryVLA 容量也充裕但塌陷),真正约束被 HC-2 + HC-7 联合承担 |
| HC-2 梯度可达 | [C8] 全可微 + [C10] | L1 + L2 | 已回应 | L2 直接给旧帧 saliency 标签 |
| HC-3 不可学 mixing | [C7] delta-rule + sg(γ) + [C2] FIFO | (结构) | 已回应(grid/bank 未满时) | 关键:γ 在 [C3][C4] 写入瞬间显式 sg |
| HC-4 aux loss | [C5][C10][C11][C12] | L2 + L3 + L4 | 已回应 | 三个独立 aux,且 stop-grad 互相隔离 |
| HC-5 边界检测器 | [C5] 连续 γ + [C10] 双信号 | L2 | **部分回应**(详见 §H Trade-off 6) | [C10] 内部"双信号"是同组件 fork,RUDDER + Jacobian 都在 [C10] 内,信噪比同步崩溃风险存在;真正彻底回应 HC-5 需要 ensemble 多个独立 saliency 估计器 |
| HC-6 实时性 | [C2][C6][C7][C8] | (推理无 loss) | **勉强满足**(详见 §H Trade-off 3) | 实测预算 90-100 ms on H100 batch=1,A100 上接近 110 ms,可能溢出 |
| HC-7 信息精度异质 | [C3][C4][C6][C7] 双通道 | L2 分通道 + L3 + L5 | 已回应 | geo / sem 通道结构性分离 |
| HC-8 NLP 模板受限 | 整体——无单一 NLP 模板被整体复用 | — | 已回应 | RMT/Compressive/NTM 元素被抽出原子机制 |
| §3.1 因果不可能 | [C5] + [C10] + [C11] | L2 + L3 | 已回应(部署时退化) | 训练时后视监督;部署时仍是代理信号(§H Trade-off 1) |
| §3.2 几何衰减 | [C7] delta-rule + LRU 丢弃 | (结构) | 已回应(容量未满时;满后 stale 内容丢失但不衰减) | 与所有类 B 工作的根本分离点 |
| §3.3 跨步信用 | [C10] 每帧 label + [C13] 双向 Jacobian | L2 | 部分回应 | Jacobian 显存 cap 强制 Δ ≤ 64,极长依赖(>64 步)仍需 BPTT;§H Trade-off 7 承认 |
| §3.4 异质共表示 | [C3][C4][C6][C7] 双通道 + [C12] | L4 | 已回应 | slot 间通过 L_CSM 异质化 |

**未被回应 / 部分回应的约束**:HC-5(部分)、HC-6(勉强)、§3.1(部署时退化)、§3.3(Δ ≤ 64 上限)。这些都在 §H 中显式承认。

**追溯不到 HC/§ 的组件**:无,每个组件至少绑一条。

---

## §F 关键设计选择的反事实辩护

### 选择 1:Delta-rule 写入 + LRU 丢弃 vs 可学门控更新 vs 合并平均

- **X** = $M^{(t)} = M^{(t-1)} + \mathrm{sg}(\gamma_t) \cdot \Delta_t$,容量满时 LRU 丢弃整个 slot
- **X' (备选 1)** = $M^{(t)} = g_t \odot \hat M_t + (1-g_t) \odot M^{(t-1)}$,g 可学
- **X'' (备选 2,原草案)** = delta-rule + 容量满时合并平均相似 slot
- **依据**:HC-3、§3.2 引理 2 + ReMem-VLA Fig 6a;Infini-attention NLP 代数证明;**审查 Bug 1 修正:合并平均 = 等价 g=0.5,会复活 §3.2 衰减,所以放弃 X''**
- **依据强度**:**[中]**(从原草案 [强] 降级)。降级理由:(a) Infini-attention 在 NLP 的"work"是 next-token streaming,与 VLA 的 cross-attn + flow matching 串接的 readout 性质不同,跨域迁移本身有风险;(b) LRU 丢弃 stale 内容这一选择在 ML 文献无直接先例,代价是错误写入永久丢失(见 §H Trade-off 5)。
- **如果错了**:可学 g 替换在长程任务上塌陷到 baseline(预期 SR 差异 -50%);LRU 丢弃错误条目导致中长程性能逐步退化(尾部 30% trajectories 被错丢)。
- **Ablation**:用可学 g 替换 [C7] 与 LRU 丢弃 → 用合并平均替换。

### 选择 2:Geometric grid + Semantic slot bank 双通道 vs 单 bank 双 head vs 按位置分

- **X** = 独立写入触发器 + 独立 aux loss 的双通道(grid + slot)
- **X'** = 单 bank,双 head 但共享 KV/写入触发器
- **X''** = v1 路线:VLM 端 + action expert 端
- **依据**:HC-7 + MEM 实验 + 独立 aux loss 的 stop-grad 隔离需求(若共用 bank,L_HCS 与 L_CSM 在同 bank 上互相污染)
- **依据强度**:**[中]**。HC-7 的 30 vs 10 bits 数量级证据并不严格支持"两个独立 bank"——同样支持"单 bank 双 head"。"按精度分"仅是 K-of-N 可行设计之一。维持 [中],理由聚焦"独立写入触发器与独立 aux loss",而非"精度跨度"。
- **如果错了**:Ablation 显示单 bank 持平甚至更好(在控制总容量的前提下)。
- **正确 ablation 设计**:对照"单一 grid(d_g 加大到 d_g + d_s 等容量)"vs"grid + slot",在控制总容量前提下消融——非"两种切分方式互比"。

### 选择 3:Hindsight Causal Saliency 监督 ESPC(L_HCS)

- **X** = γ_*_t 由 [C10] 离线 Jacobian + RUDDER 监督
- **X'** = γ_*_t 自由学,仅由 L1 + L_PRH 约束
- **依据**:§3.1 + §3.3 hindsight-as-loss + RUDDER/Mesnard 文献先例
- **依据强度**:**[弱]**。**最大赌注**。v1 §7 的 open question:"Jacobian saliency 在真实 expert demo 信噪比是否够"未被验证。RUDDER 是 RL adaptation。
- **如果错了**:γ̂ 与 γ 的 BCE 收敛但 SR 不动。Ablation:γ̂ 替换为均匀 0.5 baseline。
- **Sub-decision 3a:Base policy 选择**:E1 第一轮 base = π0.5(避开 chicken-and-egg);若 IoU > 0.4 通过,E2/E3 阶段切到 CHIME early checkpoint(信号变锐,但承担 chicken-and-egg)。
- **Fall-back(关键)**:若 E1 失败(IoU < 0.4),disable [C10][C12][C13] 与 L_HCS、L_CSM,fallback 到简化版本——架构降级为"只打 HC-3 + §3.2 + §3.4 + HC-7,放弃 §3.1 后视回应"。**这一退路是 publishable**(回应 4/8 条 HC + 3/4 条 §,workshop paper 级别 claim)。

### 选择 4:Event Segmentation 预测误差作为 ESPC 主信号

- **X** = ESPC 用 prediction error e_*_t 初值,L_HCS 校准
- **X'** = Mem-0 路线:learned classifier
- **X''** = MemoryVLA 路线:top-k retrieval similarity
- **依据**:HC-5 + Mem-0 GT vs learned 28.5→45.3 + Baldassano 2017
- **依据强度**:**[中]**。Baldassano 是 fMRI 观测者实验,与 VLA 控制 backbone hidden state 不同 substrate;假设"hidden state 预测误差形态在 VLA 中也对应 event boundaries"未被独立验证。
- **如果错了**:e_*_t 在视觉变化大但任务无关(distractor)场景上误触发,grid/slot 被无关写入污染。
- **Ablation**:在 distractor-heavy 数据上检查 γ 误触发率。

### 选择 5:Reverse Jacobian 双向计算(原草案 reverse replay 的修正版)

- **X** = [C10] 双向 Jacobian:同时算 forward J^{(t,t+Δ)} 与 reverse J^{(t,t-Δ)}
- **X'** = 仅 forward
- **X'' (原草案)** = "反向送入网络"——审查 Bug 8 揭示在 causal Transformer 下无定义,放弃
- **依据**:Foster 2017 + §3.3 信用分配 intuition——某些任务存在反向因果(柔性接触)
- **依据强度**:**[弱]**。纯 design intuition,且我自己估计反向因果场景在 VLA 里很罕见(<5% trajectories),收益边际。
- **如果错了**:开 / 关 [C13] SR 无差异。
- **赌注承认**:这是第二大赌注,代价低(只是额外一次 Jacobian),不留它则纯 forward。**MVP 默认砍 [C13]**(与 §0.7.4、§I.2 砍除清单一致,不再是"资源紧张时"的可选项)。

---

## §F.6 Ablation 预测表

(数字为预期 SR 变化,需小规模实验确认)

| Ablation | 短程 SR (<50) Δ | 中程 SR (50-150) Δ | 长程 SR (>150) Δ | 信号强度 | 直接回答的判断 |
|---|---|---|---|---|---|
| 删 L_HCS,γ_* 仅 self-supervised | ~0% | -5% | -25% | 强 | §F 选择 3 |
| 删 L_PRH | ~0% | -3% | -15% | 中 | HC-4 必要性、§3.1 代理 |
| 删 L_CSM | ~0% | -2% | -10%(slot collapse) | 中 | §F 选择 2 + §3.4 |
| 用可学 g 替换 [C7] delta-rule | -2% | -10% | -50% | 强 | §F 选择 1(HC-3、§3.2) |
| 用合并平均替换 LRU 丢弃 | ~0% | -5%(渐 §3.2 衰减) | -20% | 中 | Bug 1 修正必要性 |
| 删 [C13] reverse Jacobian | ~0% | ~0% | -3% | 弱 | §F 选择 5 |
| 合并 grid + slot 为单 bank(等容量) | ~0% | -5% | -15% | 中 | §F 选择 2 |
| 删 Attention entropy 正则(L_aux) | ~0%(无 collapse) 或 -5%(collapse) | ~ | ~ | 弱 | Bug 7 对策必要性 |
| **γ_const = 1.0 vs γ from [C5]**(γ 必要性) | ~0% | -2% / 持平 | -5% / 持平 | **关键** | **[C5] 补充 §F 辩论:γ 是否必要** |
| **γ_const = 1.0 vs γ from L_HCS**(full 版 γ 必要性) | ~0% | -3% | -8% | 强 | **若差 < 5% → 删 [C5][C10] 全套** |

**最高 ROI ablation 排序(优先跑)**:
1. 删 L_HCS(回答最大赌注)
2. **γ_const=1 vs γ from [C5]**(回答"γ 这个软门控是否冗余",见 [C5] 补充 §F)
3. 用可学 g 替换 delta-rule(回答 §3.2 claim)
4. 用合并平均替换 LRU(回答 Bug 1 修正)
5. 单 bank vs 双通道(回答 HC-7 / §3.4 设计选择)

---

## §G 训练范式与时间维度展开:任务"打开抽屉,放入红块,关上抽屉"

设 T=100 步。子任务边界:open(0-30)、place(30-70)、close(70-100)。

### t=20(open 阶段中段)

- **观测 o_20**:RGB 中可见手 + 抽屉(还未打开)、proprio = 手腕位姿
- **[C1]**:产生 h_20
- **[C2] FIFO**:M_work 已含 h_{13:20}
- **[C5] ESPC**:e_geo_20 中等(手在加速移动)、e_sem_20 低(子任务进度无突变)。γ_geo_20 ≈ 0.6,γ_sem_20 ≈ 0.15
- **[C3] Geo 写头**:中等强度(sg(γ_geo)=0.6)写入手部 + 把手附近 voxel
- **[C4] Sem 写头**:几乎不写入
- **[C8] 读出**:cross-attn 读 M_work + M_sem(几乎空)+ M_geo grid 的 coarse 层
- **[C9]**:a_20——继续向把手移动
- **L 计算**(训练时):
  - L1:flow matching loss on a*_20
  - L2:[C10] 算 ∂a*_36/∂o_20(Δ=16,在显存预算内)→ γ̂_geo_20 ≈ 0.7。L_HCS 推 γ_geo 上升
  - L3:m_20 预测 o_36 / a*_36(正在打开抽屉)
  - L4:CSM 抽 4 个 slot,M_sem 几乎空,L_CSM 信号弱
- **被更新参数**:LoRA on [C1], [C9];[C3][C4] 写头;[C8] cross-attn(经 sg 对 query 不影响 [C1]);[C5] ESPC

### t=60(place 阶段后段,抽屉**仍打开但被手臂半遮挡**)

- **观测 o_60**:红块刚被放下,手开始撤回;抽屉部分被手臂遮挡
- **[C5] ESPC**:e_sem_60 高(子任务"放入"接近完成)→ γ_sem_60 ≈ 0.85;e_geo_60 中等 → γ_geo_60 ≈ 0.7
- **[C4] Sem 写头**:**强写入**——softmax 选中低 ‖M_sem_i‖ 的 slot(假设 #18),delta-rule 累加 0.85·w_18·φ(q_60)⊗v_60。Slot #18 编码"红块已置入抽屉"
- **[C3] Geo 写头**:写入红块当前位置(抽屉内坐标)
- **[C8] 读出**:M_sem 已含 slot #18 + 早期 t=30 的"抽屉打开"(slot #5)
- **[C9]**:a_60 = 撤回手
- **L 计算**:
  - L2:[C10] 算 ∂a*_76/∂o_60(Δ=16)→ γ̂_sem_60 ≈ 0.85。**Δ=64 时算 ∂a*_98/∂o_60**(若 batch 显存允许),会捕获"关到底"对"红块在内"的远期依赖,γ̂ 可能 ≈ 0.95
  - L3:m_60 预测 o_76 / a*_76(开始关抽屉)
  - L4:CSM 抽 4 slot,KL 此时不大(因为 t=60 当前动作 = 撤手与 slot #18 因果联系弱;但在 t=85+ 时 KL 会大,见下)
- **关键**:slot #18 此后**永不被衰减**(delta-rule 加性);除非容量满 + LRU 丢弃,否则保留。Episode 长度 100 远小于 K_s=64,LRU 丢弃在此 episode 不触发

### t=90(close 阶段后段)

- **观测 o_90**:抽屉接近关闭,**红块完全不可见**
- **[C5] ESPC**:γ_sem_90 ≈ 0.4
- **[C8] 读出**:cross-attn 从 M_sem 读出——slot #18(因 q_90 与 k_18 高余弦相似)+ slot #5。Action expert 知道"抽屉里有东西"
- **[C9]**:a_90 = 继续关到底,不需要重开
- **L 计算**:
  - L1:正常 BC
  - L4:**关键时刻**——KL(π(a|m_90) ‖ π(a|m_90 \ slot #18))**应当大**。如果忘了 slot #18,策略可能去重开抽屉。**这是 L_CSM 推 [C4] 把"红块入抽屉"事件正确路由到独立 slot 的关键监督信号**
  - **L2 反向到 t=60**:在 t=90 这个 mini-batch 元素中,[C10] 计算 ∂a*_90/∂o_60 时 Δ=30,**超出默认 Δ ∈ {4, 16}**——此时若 batch 显存允许,启用 Δ=64 capture;**若不允许**,这条远程依赖只能由 L_PRH 间接承担(m_60 预测 o_{60+64}=o_124 越界 → 退化为预测 o_76,长程信号丢失)。**这是 §H Trade-off 7 显式承认的代价。**
- **被更新参数**:[C5][C8][C9] LoRA、[C4] MLP_q/v(via L_CSM)

---

## §H 吞下的 trade-off

### Trade-off 1:部署时 ESPC 仅有代理信号,后视监督在部署时不可得

**何时获得**:训练时 [C10] 把"未来需要"推入 γ_*_t,[C5] 学到对 prediction error 的加权近似。

**代价**:部署时只有 self-supervised prediction error。Test 任务的 e 分布偏移会导致 γ_*_t 失准。

**何时可接受**:训练数据覆盖 test 视觉 / 语义分布(域内 evaluation)。
**何时不可接受**:跨 dataset zero-shot;对抗 distractor。

### Trade-off 2:不可变 bank 不能"忘掉错误信息"——错误写入永久污染

**何时获得**:§3.2 几何衰减不发生(容量未满时)。

**代价**:ESPC 误触发的内容被永久写入,delta-rule 加性使其无法衰减;LRU 丢弃只丢"最不重要"的 slot,不能纠错——错误内容若被 [C12] CSM 错误地评为"重要",反而更难被丢弃。

**何时可接受**:training [C10] 准确,误触发率 < 5%;episode 长度 < bank 容量。
**何时不可接受**:在线终身学习。**CHIME 在 episode 间清空,显式回避**——也放弃了 cross-episode 学习,与 v1 §5.3 一致。

### Trade-off 3:HCS Jacobian 计算成本 + 训练-推理代码路径分裂

**何时获得**:§3.3 信用分配通过离线监督而非 BPTT。

**代价**:训练 pipeline 比纯 BC 慢 ~2-3x。**纸面估算**(H100 BF16 batch=32,utilization 35%——非实测,需在目标硬件上验证;6×A800 + batch=24 的对应估算见 §0.7.1):
- 单 step 主路径 forward+backward:140 GFLOPs
- L_HCS Jacobian × 3 个 Δ:420 GFLOPs(若 Δ=64 含,显存边缘)
- L_CSM 4 slot leave-one-out × 2 frozen forward:400 GFLOPs
- L_PRH 3 head:15 GFLOPs
- **总单 step ≈ 1045 GFLOPs ≈ 800-1000 ms**
- Epoch 时间:10M frames / batch 32 = 312k steps × 0.8s ≈ 70 h/GPU
- **4×H100 DDP(85% 效率):20 h/epoch,1 周训练 = 5-6 epoch**(非原草案的 8 epoch)
- **OOM 风险 ~30%**:Δ=64 + base policy activation 跨 64 步 ≈ 30 GB at fp16,接近 H100 80GB 上限。Fall-back:Δ ∈ {4, 16}。

**部署时不跑这些**,但 train-only / deploy 路径分裂是工程负担——需要严格 conditional forward 与 checkpoint 分组。

**何时可接受**:≥ 4×H100,训练预算 1 周;开发流程严格区分 train / deploy 路径。
**何时不可接受**:单 GPU、训练时间敏感场景。**fall-back**:简化到 [C5][C2][C6][C7][C8][C9] + L1 + L_PRH 的 MVP 版本。

### Trade-off 4:Slot bank 容量 K_s=64 在超长任务(>500 步)上不够

**何时获得**:HC-1 容量在 200-step 任务上充裕。

**代价**:K_s=64 对应 ~64 events。500-step 厨房任务的 event 数可能 80-100 → 触发频繁 LRU 丢弃(每次丢 1 个 slot 整体置零),长程 SR 渐降。

**何时可接受**:目标任务 ≤ 200 步。
**何时不可接受**:Goal2Skill 厨房分钟级任务、家务多房间任务。需要 K_s ≥ 128。

### Trade-off 5:LRU 丢弃 vs 合并平均 vs 可学 g —— 三选其一,没有完美选项

**何时获得**:LRU 丢弃保留 delta-rule 代数性质(Bug 1 修正)。

**代价**:被丢弃的 slot 内容**彻底不可恢复**。如果 [C12] CSM 评分错误(把任务相关 slot 评为低重要度),LRU 丢的就是关键事件——比合并平均的"信息模糊化"在某些任务上更糟。

**何时可接受**:K_s 容量大于实际 event 数 1.5×(64 vs 40 events)。
**何时不可接受**:event 密度高 + 长 episode。

### Trade-off 6:HC-5 部分回应——[C10] 自身仍是单点

**何时获得**:γ_*_t 不再单一来自 learned subtask classifier(避开 Mem-0 模式)。

**代价**:**[C10] HCS-H 内部"双信号"(Jacobian + RUDDER)实际是同一组件 fork**——若 RUDDER LSTM 在 1k-10k 轨迹上欠定 + base policy Jacobian 噪声大,γ̂ 就是噪声,与 Mem-0 learned classifier 在结构上同型。HC-5 的本质是"避免依赖单一组件",这条只是把"分类器单点"换成"saliency head 单点"。

**何时可接受**:E1 通过(IoU @ 0.3 ≥ 0.4),信号质量已实证。
**何时不可接受**:E1 失败 → Trade-off 6 升级为"完全没回应 HC-5",必须 fallback 到 §F 选择 3 简化版,显式放弃 HC-5 的回应。

**未来扩展**:引入第二独立 saliency 估计器(例如 occlusion-based gradient × input)做 ensemble,任何单一估计器训坏不致 γ̂ 全失。MVP 不做。

### Trade-off 7:§3.3 信用分配的 Δ 上限

**何时获得**:每帧 saliency label 绕过 BPTT 跨百步反传。

**代价**:Jacobian 显存 cap 强制 Δ ≤ 64(默认 {4, 16},显存允许时加 64)。**任务的真实因果跨度若 > 64 步**(例如 200-step 任务的"开始 vs 结束"依赖),L_HCS 仍无法直接监督——只能间接靠 [C13] 反向 Jacobian 与 L_PRH 长 k 间接承担。

**何时可接受**:任务的 critical 因果跨度 ≤ 64 步。
**何时不可接受**:超长任务(>500 步,跨多个 100-步以上的因果链)。

### Trade-off 8(隐式):HC-6 实时性勉强满足

**何时获得**:所有 forward 操作 GPU 并行。

**纸面估算 on H100 BF16 batch=1**(非实测,基于 FLOPs + 显存带宽推算,需实测确认;4090 48G 估算见 §0.7.2):
- [C1] SigLIP-ViT-L:30-50 ms
- [C5] ψ over 2k token:5-15 ms
- [C8] cross-attn over 2k KV(M_work + M_sem)+ N_geo_q=16 三线性采样 grid:5-10 ms(关键:N_geo_q × 8 邻居采样 = 128 voxel 读,不是遍历 5e4)
- [C9] flow matching ODE 求解:30-60 ms
- **Total ≈ 70-135 ms**

**HC-6 在 H100 上勉强 < 100 ms,在 A100/4090 可能溢出**。MVP 阶段允许溢出到 150 ms(对应 6.7 Hz 控制频率,仍在可接受窗口内)。**4090 48G 推理需要叠加 §0.7.3 列出的优化(ViT-L → ViT-B、ODE 1-step 蒸馏、INT8 量化),才能压回 70-140 ms 区间**。

---

### §H.5 系统性失败的任务结构(明确列出本架构会失败的场景)

(原草案遗漏,审查补充)

1. **Distractor-heavy 视觉变化大但任务无关**:机器人长时间巡航 / 背景人员经过 → e_geo 飙高,γ_geo 错误高 → grid 被无关几何写满。**对策**:加 distractor 增强训练数据。**未对策时,本架构会失败。**

2. **反向因果场景**:t 时刻动作 a_t 影响 t-Δ 时刻该被写入什么(柔性接触延迟反应)→ forward Jacobian 时序方向错。**对策**:[C13] reverse Jacobian。**若反向因果 > 5% 帧,需重新设计。**

3. **物体身份重排**(red↔blue 互换):slot 通过 content-based key 分配,身份混淆物体可能被路由到同一 slot。**对策**:启用 SymObj 类的物体级初始化(本文档未涵盖,见 v1 §5.2 类 D 拒绝路线)。**未对策时,失败。**

4. **极长 episode(>500 步)+ 高 event 密度**:K_s=64 不够 → LRU 频繁丢弃 → 关键事件可能被丢失。**对策**:K_s ≥ 128 + ensemble multiple bank。

5. **跨 episode 知识依赖**(如同一物体的物理属性):episode 间清空 M_sem,本架构显式不支持。**重新设计需要 cross-episode persistent slot,放弃 v1 §5.3 的 disclaim。**

---

## §I 后续工作流接口(给作者本人)

### §I.0 数据集前提(关键)

| Benchmark | 公开? | 连续 trajectory? | 平均长度 | 子任务边界标注? | E 阶段 |
|-----------|------|----------------|---------|--------------|--------|
| BridgeV2 | **是**(rail-berkeley/bridge_data_v2) | 是,raw HDF5 完整 traj | 30-50 步 | **无**——E1 需手标 100-200 traj(1-2 人周额外) | E1, E4 |
| RoboCasa | **是**(robocasa.ai) | 是,sim 端可任意长 | 50-200 | sim 端可生成 | E2, E4 |
| LIBERO-Long | **是** | 是 | 100-200 | 部分 | E5 替代 |
| CALVIN(ABCD→D) | **是** | 是 | 长 | 子任务标注完整 | E5 替代 |
| SimplerEnv-LongHorizon | **是** | 是 | 长 | 部分 | E5 替代 |
| RMBench | **未公开** | — | — | — | E5 原计划 → 替换 |
| MemoryBench(ReMem-VLA) | **未公开** | — | — | — | E5 原计划 → 替换 |
| EPIC-KITCHENS / MemER | EPIC 公开,MemER 标注未必 | 是 | 长 | 部分 | E2 备选 |

**关键风险**:**BridgeV2 没有 frame-level event 边界标注**——E1 必须先手标 100-200 traj。这是文档原草案完全没提的 1-2 人周额外工作。

### §I.1 组件清单与验证类型

| 组件 | 类型 | 验证 | 优先级 | 工作量 | 风险 |
|------|------|------|--------|------|------|
| [C1][C2][C9] | 既有 | 不必 | — | 1 人周 | 低 |
| [C3][C6] | 既有(NICE-SLAM)+ adaptation | 可独立(配 [C5] stub) | 中 | 4-5 人周 | 中 |
| [C4][C7] | 既有(Infini-attn + XMem)+ adaptation | 必须联立 [C5][C10] | 高 | 4-5 人周 | 中 |
| [C5] ESPC | **新设计核心** | 部分独立(配 [C10]) | **高** | 3-4 人周 | **高** |
| [C8] | 既有(Memorizing Transformer) | 不必 | — | 1 人周 | 低 |
| [C10] HCS-H | 新设计核心 | 可独立(E1) | **最高** | 5-6 人周 | **高** |
| [C11] PRH | 既有(MERLIN)+ adaptation | 可独立 | 中 | 1.5 人周 | 低 |
| [C12] CSM | 既有(Mesnard)+ adaptation | 必须联立 [C7] | 中 | 1.5 人周 | 中 |
| [C13] reverse Jacobian | 新设计 | 仅 ablation | 低 | 1 人周 | 中 |
| 数据加载 + benchmark adapter | — | — | 高 | 3-4 人周 | 中 |
| 整体 train loop + multi-loss debug | — | — | 高 | 4-6 人周 | **高** |
| Eval pipeline | — | — | 中 | 3 人周 | 中 |
| **总计** | | | | **~33-40 人周 ≈ 8-10 人月** | |

### §I.2 MVP(3-month)版本

> **本节已被 §0.7.4(硬件锁定 MVP 配置)替换。** §0.7.4 是 MVP 唯一基线,本节内容已删除以消除双轨歧义。详见 §J.2 修订记录。

### §I.3 6-month full 版本 milestone

- **M1 (week 4)**:E1 完成。BridgeV2 100-200 traj 手标(1-2 人周额外)+ [C10] J^{(Δ)} + IoU 报告。**门槛:IoU @ 0.3 ≥ 0.4**。失败 → fallback 到 MVP(§0.7.4 配置),λ_1 永久锁 0。**λ_1 anneal 从此 milestone 出口启动**:E1 PASS → λ_1 在接下来 5000 step 内 0 → 0.3 anneal,然后恒定。E1 SOFT-PASS (0.3 ≤ IoU < 0.4) → λ_1 anneal 起步但目标值降为 0.15,记 red flag #1 但继续。
- **M2 (week 8)**:E2/E3 完成。[C5] 在 RoboCasa 单子任务收敛 + [C11] PRH 在 1k traj 上独立收敛。
- **M3 (week 12)**:[C3][C6] + [C4][C7] 联立可训(L_main + L_PRH),写头梯度反向通畅,memory 内容非崩塌。
- **M4 (week 16)**:E4 完成。RoboCasa 双子任务 + 完整 5-loss 与 baseline ≥ 10% SR 提升。
- **M5 (week 20)**:LIBERO-Long + CALVIN 全架构跑通,E5 报告。
- **M6 (week 24)**:Ablation 套件(**§F.6 表中 10 项**,含两条 γ 必要性 ablation)+ 论文写作。

### §I.4 Red flags(早期信号 → 项目要停下来重新评估)

1. **M1 E1 IoU < 0.3**(< 门槛 0.4):整个 §F-选择 3 推翻,[C10] 必须砍,fallback 到 MVP。**作者认为这是最大赌注,工程直觉概率 30-50%**。
2. **训练第 1 epoch 后 γ_sem 塌缩到 ~0.5 ± 0.05(无方差)**:ESPC EMA 标准化吃掉所有信号。整个写入触发器失效。**对策:重新设计 EMA 系数 / sigmoid 替换为 hard threshold + soft margin。**
3. **L_PRH 在 k=64 上不下降**(只在 k=4 降):memory 实际只能缓存 ~10 步,长程能力为零,§3.2 claim 实证伪。**对策:回退到 [C7] 容量上限增加 + delta-rule 系数调整。**
4. **slot bank softmax 退化为 hash collision**(top-1 slot 占 80%+ 写入):L_CSM 没生效。**对策:加 entropy regularizer 或 routing trick(Switch-Transformer 风格 load balance)。**
5. **OOM at batch 16**(预期 batch 32):[C10] Δ=64 显存撑不住。**对策:Δ 限定 {4, 16},接受 §H Trade-off 7 升级。**
6. **多 loss balancing 第 8 周仍在 sweep**:5-loss 之间 conflict 严重,工程不可控,论文 claim 难立。**对策:用 GradNorm 或 PCGrad 自动平衡。**

### §I.5 不会做的事

- 不会基于这份架构整体训一个完整 VLA。组件没通过 E1-E4 之前不进入 E5。
- 不会在 E1 失败后强行 patch [C10]——直接 fallback 到 MVP。
- 不会在 BridgeV2 之外的 dataset 上跑 E1(其他 dataset 没有连续 trajectory + a*_{t+64} 对齐保证)。

---

### §I.6 工程决策锁定表(v2.1 新增)

> 本表是 v2.1 一致性补丁的核心:把"可以推出但文档没明说"的工程契约**显式锁定**,消除写代码时被迫"猜决策"的歧义。每行一锁。

| # | Decision | Lock |
|---|---|---|
| 1 | **stop-grad 矩阵** | 7 entries,见 §B.1 锚点 `<!-- SG-MATRIX-CANONICAL -->`;CI 强制 `tests/test_grad_flow.py` 全绿 |
| 2 | **数据 I/O** | LIBERO-Long h5 → per-episode `.pt` cache(继承 Hindsight CODE_STANDARDS §1.1);schema:`rgb_feature: fp16 [T,1536] / proprio: fp32 [T,8] / action: fp32 [T,8] / sub_task_id: int32 [T] / episode_id: int` |
| 3 | **训练超参** | `lr=1e-4 cosine`,`bs=24`(per-rank,DDP),`grad_clip=1.0`,`warmup=500 steps`,`betas=(0.9, 0.95)`,`wd=0.01`,`precision=bf16-mixed`(saliency 强制 fp32);grad accumulation = 1(MVP) |
| 4 | **Memory tensor batch 维** | `M_geo: [B, L, D, H, W, C_g]`(L=级数,MVP=1);`M_sem.v: [B, K_s, d_s]`,`M_sem.k: [B, K_s, d_s]`,`slot_free: [B, K_s]`;`M_work: [B, K_w, N, d_h]`;episode 边界由 `reset_memory(batch_idx)` callback 显式清零 |
| 5 | **Forward 顺序** | per §B.2 伪代码:`C1 → C5 → C2.append → {C3, C4} → C8 → C9 → loss`(C5 必须在 C2.append 之前看 `M_work^{t-1}`) |
| 6 | **Loss reduction** | 全部 `mean over (B, valid-T)` with episode mask;`masked_mse`(继承 `Hindsight/src/utils/losses.py`)即此约定 |
| 7 | **λ_1 schedule** | per §D 修订:E1 通过前 = 0;通过后 5k step 线性 anneal 0 → 0.3;之后恒定 0.3 |
| 8 | **§I.2/§0.7.4 双轨** | per §I.2 已折叠为单行,§0.7.4 是 MVP 唯一基线 |
| 9 | **HCS-H 实现路径** | offline batched script;产物文件协议:`Hindsight/output/saliency/gamma_hat/per_task_q75/libero_long/ep_NNNNNN.pt` 含 `{gamma_geo: [T], gamma_sem: [T], meta: {...}}`;CHIME-VLA 通过 `chime_vla.hindsight.consumer` 加载 |
| 10 | **L_CSM 触发频率** | 每 mini-batch,从 K_s=64 slot 中均匀抽 4(架构 line 1462);per-step 触发,每 step 多 4 次 frozen [C9] forward(~6% 训练开销) |
| 11 | **slot_free mask** | per [C7] 修订:外部 bool mask + softmax logit penalty (减 1e9);单测 `tests/test_slot_lifecycle.py` 验证 |
| 12 | **Config 系统** | Hydra + structured dataclass;镜像 `Hindsight/src/config.py` 模式;`@dataclass class ChimeConfig` 含 60-80 字段(详见 IMPLEMENTATION_PLAN.md) |

**已知留作实验决定**(不在锁定表内):
- α_a (L_PRH 中 action 分量权重) = 1.0 [标定值,M2 后 sweep]
- EMA warmup 系数曲线 [M2 训出再回填]
- N_q / N_geo_q = 16 [标定值,M3 后 sweep]
- Δ ∈ {4, 16}(64 仅在 A800 显存允许时启用,M3 后)

---

## §J 审查痕迹与放弃的子主张

(本节透明记录:文档已通过三轮独立 adversarial review 收敛;以下是审查中被点中、最终放弃或显式降级的子主张。)

| 子主张(原草案) | 审查发现 | 处理 |
|---|---|---|
| "选择 1 [强]:数学+实证支持" | Infini-attention 是 NLP 域,VLA 跨域迁移有风险 | 降为 [中] |
| "[C7] 合并平均做 consolidation" | 等价 g=0.5 mixing,破坏 §3.2 代数 | 改 LRU 丢弃 |
| "HC-5 已回应" | [C10] 内部"双信号"是同组件 fork,仍是单点 | 降为部分回应,§H Trade-off 6 显式承认 |
| "[C10] 通过离线 Jacobian 绕过 BPTT" | Jacobian 仍需跨 64 步 frozen forward,显存 cap 强制 Δ ≤ 64 | §H Trade-off 7 + Δ 默认 {4, 16} |
| "stop-grad on h_t 阻断 [C1] 污染" | 经 [C8] query 路径仍泄漏 | 显式 sg query 投影矩阵 |
| "K_w·d_h + 5e4·d_g + K_s·d_s ≈ 5MB" | 算错(应 ≈ 12.9 MB at fp32) | 修正,且重新表述 HC-1 narrative |
| "[C2] FIFO 不会被训成短视" | 读端 cross-attn 可学,仍可被训成短视 | 加 attention entropy 正则 L_aux |
| "[C13] 反向送入网络" | causal Transformer 下无定义 | 改为反向 Jacobian(数学一致) |
| "v1 判断四:语义 vs 感知信号分布密度模式" | finer-grained claim 被简化丢失 | §A 第二条差别明确说明:简化为同 ψ + 双 projection,fallback 是双 ψ |
| "数据集 BridgeV2/RMBench/MemoryBench" | RMBench/MemoryBench 未公开;BridgeV2 无事件标注 | §I.0 显式列出,加 1-2 人周手标 |
| "E1 失败 → 重新评估架构" | 与 §F 选择 3 fall-back 不一致 | §I.4 明确 fallback 到 MVP,不阻塞 E2-E5 |

### §J.2 第四轮一致性审查后补丁(2026-04-28)

经独立 audit 发现 14 处累积编辑产生的内部不一致,以下是对前 10 处的修复:

| 不一致 | 位置 | 修复 |
|---|---|---|
| LRU 丢弃名称冲突([C6] timestamp vs [C7] CSM-importance 同名"LRU") | [C3] step 3、[C7]、§A 第三条 | 把 [C6] 的"timestamp LRU"改名为"timestamp eviction",保留"LRU"专指 [C7] CSM-importance-based |
| [C5] "已 benchmark"措辞遗留 | [C5] 接口契约 | 改为"5-15 ms 在 H100 上(纸面估算口径,需实测确认)" |
| [C5] latency "< 8 ms" vs "5-15 ms" 数字冲突 | [C5] 接口契约 | 统一为 5-15 ms |
| t=60 γ_geo 数值缺失 / 不一致 | §G | 补 γ_geo_60 ≈ 0.7 与 [C3] 补充演示对齐 |
| §I.2 MVP 与 §0.7.4 MVP 两套并存 | §I.2 头部 | 显式声明 §I.2 已被 §0.7.4 替换,保留作为 H100 假设对照 |
| [C10] L_HCS sg list 不完整 | [C10] 梯度流向 | 改为"不流入 [C1][C3][C4][C6][C7][C9]"与 §D L2 对齐 |
| HC-1 §E 状态列措辞漂移 | §E 表 HC-1 行 | 改为"已回应(结构充裕,非真正瓶颈)" |
| §0.6 与 §0.3 "双信号"措辞遗留 | §0.6 PSM-CWM 行、§E 表 HC-5 行 | 统一改为"双 fork 信号(同组件内部 fork)" |
| §F 选择 1 [中] vs §0.1 "代数事实"强口径 | §0.1 段 | 改为"代数趋势"+ 引用 §F 选择 1 [中] 强度 |
| [C13] MVP 砍除条件性 vs 默认 | [C13] §F 选择 5 末 | 改为"MVP 默认砍 [C13]" |
| §I.3 M6 "§F.6 表中 8 项" vs 实际 10 项 | §I.3 M6 | 改为"10 项,含两条 γ 必要性 ablation" |
| §A 第三条未提 grid 也有 LRU | §A 第三条 | 补"slot bank 用 CSM-importance LRU、grid 用 timestamp eviction" |
| §H Trade-off 3/8 "实测预算" 表述 | Trade-off 3 / 8 表头 | 改为"纸面估算"(非实测),加目标硬件验证提示 |
| §I.2 / §0.7.4 MVP 不回应 §3.3 未明示 | §I.2 砍除清单后 | 加注 "MVP 不回应 §3.3,与 §F-选择 3 fall-back 一致" |

**未修复的 1 处**(影响最小,留档):§F.6 ablation 排序 #2 "γ_const=1 vs γ from [C5]" 在 §I.3 milestone M1-M6 中没有具体对应 milestone。这条 ablation 在 M6 的"Ablation 套件"统称里包含,但未单独列时间——若读者要按 milestone 跑,需自己拆分。

---

### §J.3 v2.1 一致性补丁(2026-05-07)

经第 5 轮审查(4 个并行 agent)发现 v2 与 chime_vla_proposal 之间有 5 处刚性矛盾 + 12 项工程契约缺口。本版本(v2.1)统一,改动如下(每条都不影响架构本体,仅消除歧义):

| # | 位置 | 修订 | 原因 |
|---|---|---|---|
| **D1** | §B (line 344-348) sg 列表 | 4 条 → 7 条 canonical 表 + `<!-- SG-MATRIX-CANONICAL -->` 锚点 + §B.1 CI 单测契约 + §B.2 forward 顺序伪代码 | proposal §5.4 列 7 处,architecture 仅 4 处;实施期会漏 3 条 sg |
| **D2** | §I.2 | 整段折叠为单行"已被 §0.7.4 替换" | §J.2 line 2035 已声明替换但条文并存,双轨容易误读 |
| **D3** | proposal §B.3 (该文件) | MVP 保留列表补 [C11] | L_PRH 是 MVP 主信号,必须保留 [C11] head |
| **D4** | §D total loss + §I.3 M1 | λ_1 由静态 0.3 改为 schedule(E1 前 0,后 5k step anneal 0→0.3,然后恒定) | proposal §7 line 313-325 隐式课程论证;静态 0.3 会撞 cold-start |
| **D5** | [C7] + §B.2 forward 顺序 | 增 `slot_free ∈ {0,1}^{B×K_s}` 外部 mask + softmax logit penalty(写入与读出端均用) | proposal §2 line 65 提示但 architecture 无落地;cos(q,0) 数值未定义 |
| **§I.6** | §I.5 之后(新) | 12 项工程决策锁定表 | 数据 I/O / 训练超参 / batch 维 / forward 顺序 / loss reduction / Hindsight 文件契约 等全文档零规约 |

**审查 agent 报告位置**:plan file `/home/sqmluser/.claude/plans/code-structure-md-agent-agent-woolly-crane.md` 的 Stage 1 章节列出了完整诊断。

---

**致读者**:本文档欢迎被进一步推翻;推翻请引用具体组件编号 + 引用 HC/§ + 给出反证或更优替代。审查痕迹保留可见,作为后续讨论的接力棒。
