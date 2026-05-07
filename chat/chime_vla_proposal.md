# 长程 VLA 的双通道不可变记忆：CHIME-VLA 方案构想

> **写作风格说明**。本文按"先讲为什么，再讲是什么"的次序展开。每个模块都由前一个模块或前一个约束逼出来，而不是平铺直叙地一个个介绍。涉及未定项时，给出候选 + 各自利弊 + 倾向，但**不强行收敛**——这些点就是当前还想不死的地方，给出"为什么不易收敛"比硬给答案更诚实。
>
> 术语先约定：
> - **HC-x**：从 problem statement 继承的 8 条 hard constraint（容量、梯度可达、不可学 mixing、aux loss 必要、边界检测器单点、实时性、信息精度异质、NLP 模板受限）。下文复用编号。
> - **§3.x**：原 problem statement 的四个不可能性论证（因果不可能、几何衰减、跨步信用、异质共表示）。
> - **delta-rule**：M_t = M_{t-1} + Δ_t，**没有** (1−g)·M_{t-1} 那个乘性衰减项。这是与门控 RNN 的代数分水岭。
> - **画布 / grid**：按 3D 物理坐标索引的 voxel grid，下文写作 M_geo。
> - **slot bank**：按 content-based key 索引的事件槽 bank，下文写作 M_sem。
> - **γ**：每帧由 ESPC 输出的两个 ∈[0,1] 软门控标量（γ_geo、γ_sem），乘在写入公式最外层。
> - **hindsight saliency**：离线已知未来时反推出的"这帧值不值得记"的真值，记作 γ̂。

---

## § 0. 我们到底要解决什么问题

VLA 的动作头是短窗的——它看到当前帧、可能加一个滑动短窗，然后回归未来若干步动作。在十几秒的子任务里勉强能用，但只要任务横跨多个语义阶段（"开抽屉 → 把红块放进去 → 关抽屉"），机器人**已经做过什么**就成了关键先验，而短窗动作头从结构上看不到这种先验。

直觉上的解法是把过去 N 帧塞进上下文。这条路用 self-attention 撞二次复杂度的墙，且更根本的问题是：**历史的"原始形式"不适合作为决策的输入——太大、密度太低、太多细节是无关的视觉外观。** 但更有意思的是，即便把历史压缩之后，长程记忆要 work，仍然要同时穿过四个针眼，这四个针眼是从问题侧反推出来的：

- **HC-3 / §3.2："可学 mixing"是结构性绝路。** 一旦记忆更新写成 M_t = g·M̂_t + (1−g)·M_{t-1}，常规 g ≤ 0.5 设置下旧内容跨 50 步乘 (1−g)^{50} ≈ 10^{-15} 量级——belief state 在结构上就保不住了。这不是调参问题，是代数事实。**意味着记忆更新必须是加性的（delta-rule），且任何与 mixing 相关的乘子都不能由主任务 loss 训。**
- **HC-2 / §3.3：跨百步 BPTT 不可达。** 哪怕梯度图保留下来，反传 100+ 步在显存和 vanishing 上都是绝路。**意味着旧帧的"是否值得记"必须有一条直接梯度路径，不能依赖端到端 BPTT。**
- **§3.1：写入时无法预知未来需要什么。** 在线时刻 t 不知道未来 200 步要什么，自监督代理（"预测误差大就重要"）没有理由对齐"未来需要"。**意味着主监督信号必须在离线已知未来时反推（hindsight），才能从根本上对齐目标。**
- **HC-7 / §3.4：几何（亚厘米）与语义（任务进度）相差三个数量级，共表示有损。** 把它们塞进同一组 latent，要么几何被语义平滑掉、要么语义被几何细节淹没。**意味着必须按精度分通道，不是单 bank。**

这四条互相绑定，下面的每个模块都是被这四条里的某一条逼出来的。这跟"挑一个先进结构来试试"完全不是一回事——是**约束在挑模块**，不是模块在挑约束。

**为什么不是六条而是四条**——HC-1（容量）和 HC-8（NLP 模板受限）在所有现代 VLA 上都不是真正的瓶颈：HC-1 容量算下来核心 buffer 大约 9-13 MB（fp32：~3.6 万 voxel × 64 dim × 4B = 9.4 MB 几何 + 64 slot × 256 dim × 4B × 2 (key+value) = 0.13 MB 语义 + FIFO 8 帧 × 256 token × 1152 dim × 4B 约 9.4 MB working buffer 视复用情况——精确数字依赖 KV-cache 复用与 fp16/fp32 选择，H_required ~100 bits 下界远低于此），结构上充裕；HC-8 是"不要整体复用 NLP 模板"的负面约束，靠拒绝架构选项而非主动设计来满足。§3.3（跨步信用分配）则与 HC-2 同构——两者都指向"旧帧需要直接梯度路径"，所以并入 HC-2 一起处理。**真正驱动设计的就是上面这四条针眼**——其它约束要么结构充裕（HC-1）、要么靠拒绝代替（HC-8）、要么并入主针眼（§3.3）。

---

## § 1. 整套方案的因果链一次讲完

把整套设计预览一下，但**模块名只在它被前一节的限制逼出来时才出现**——这样读者读 TL;DR 就能跟上 motivation 链，而不是先记一堆名字再回头找理由。

HC-3 + §3.2 直接判死可学 mixing，逼出**delta-rule 加性更新 + 不可变 bank**——bank 容量未满时代数上零衰减，满了走 LRU 整体丢弃而不是合并平均（合并平均等价 g=0.5，会复活 §3.2 衰减）。这是与所有"门控 / EMA / 写门可学"那一系工作的根本分离点。

HC-2 + §3.3 判死跨百步 BPTT，逼出**离线 hindsight saliency**——离线时未来已知，反推每帧"值不值得记"的真值 γ̂，把它当 BCE target 训写入门，旧帧的梯度路径变成"每帧一个 label"而不是"跨百步反传"。这是 §3.1 的唯一突破口（在线无法预知未来，但训练时离线可以）。

HC-7 + §3.4 判死单 bank 共表示，逼出**双通道**——几何走按 3D 坐标索引的 voxel **画布 M_geo**（亚厘米精度，适合存"哪个位置有什么"），语义走按 content key 索引的 64 槽 **slot bank M_sem**（事件级，适合存"发生过什么事件"）。两通道独立写入触发器、独立 aux loss、独立读出 query。

但 hindsight saliency 在部署时不可得（部署时未来还没发生），所以**写入门本身**必须在部署时也能跑——逼出 **ESPC（Event-Segmentation Prediction-error Controller）**：用一个轻量 ψ 在 M_work 上预测 ĥ_t|t-1，把 ĥ 与 h_t 的预测误差按几何 / 语义两条投影分别归一化为 γ_geo / γ_sem ∈ [0,1] 软门控。**γ 是连续标量而非 0/1 分类**——这是回应 HC-5 的关键（单一分类器误差被任何上限封顶；连续 soft signal 不会）。训练时 L_HCS BCE 把 γ 推向 γ̂，部署时 γ 自跑（hindsight 影响沉淀在 ψ 参数里）。

写入门 + 双通道 + 不可变 bank 还不够——HC-4 要求 aux loss 给写入端独立反传压力（不能让 L_main 单独支撑写入门，那样会被 BPTT 截断 + sg 隔离反复掐到接近零）。逼出三个独立 aux：**HCS-H**（hindsight saliency target，回应 §3.1 + HC-2）、**PRH**（predictive read head，要求 m_t 对未来 k 步可预测，回应 §3.1 代理）、**CSM**（counterfactual slot mask，leave-one-out 强制 slot 异质化，回应 §3.4）。三个 aux 之间靠 stop-grad 矩阵互相隔离——任何一处 sg 不一致整个 HC-3 隔离失效。

最后是读出端。M_work（短期 FIFO）+ M_sem（slot bank） + M_geo（grid）三路 KV 拼接给 [C9] action expert——两路用 cross-attn，几何路用 spatial query 三线性采样以满足 HC-6 实时性。读端 cross-attn 是可学的，会被训成"只读最新一帧"的短视退化路径——加 attention entropy 正则做最低限度的对策。

读完后续章节会明白：上面每个名字的引入都是被前一段抛出的约束拽出来的，拿掉任何一个会让前一段的问题重新无解。整体架构不是"叠了多少 trick"，是**每个 trick 在堵哪个针眼**。

---

## § 2. 为什么是 delta-rule + 不可变 bank，不是门控更新

直接讲为什么所有"先进的 RNN-like 写门"都过不了这一关。

**门控更新的代数趋势。** 类 B 工作（ReMem-VLA、各种 EMA-mixing、可学门控写头）的更新形式是 M_t = g·M̂_t + (1−g)·M_{t-1}。哪怕 g 完全准（每次都对），常规设计下 g ∈ (0, 0.5) 区间——因为 g > 0.5 等于"主要看新内容"，写入瞬间就抹掉历史。50 步后旧内容剩下的量级是 (1−g)^{50}：g=0.3 时 ≈ 10^{-8}，g=0.1 时 ≈ 10^{-2}——前者直接归零、后者贴着噪声本底。**这不是"训练得不够好"，是更新规则的代数后果。** 任务横跨 100-200 步时这件事直接决定了它能不能保住 belief state。

**delta-rule 把 (1−g) 这一项删掉。** 更新形式变成 M_t = M_{t-1} + Δ_t——加性，没有乘子。带来的代数性质是：**没人写它的时候它字面意义上不变**。voxel (9,9,12) 在 t=31 之后没有任何 token 投到这个位置 → `M_geo[(9,9,12)] += 0` → 内容保持 t=30 时的累积值。跨 170 步零衰减不是某个机制保住的，是**根本没有"稀释"这个动作存在**。

**但 delta-rule 不能让 mixing 项可学。** 如果让 sg(γ_t) 中的 γ 由 L_main 训，BPTT 截断后 γ 会被训成全 1（短视：让当前帧拿到最大权重）——HC-3 的本质就是这件事。所以 γ 必须由独立信号源训（hindsight saliency / prediction error），且**写入瞬间必须 sg(γ)**——这是后面所有 stop-grad 矩阵的起点。

**容量满怎么办——LRU 丢弃，不合并平均。** delta-rule 的代数性质只在容量未满时成立：bank 满后必须做点什么腾位置。原草案做的是"相似度 > 0.9 合并平均"，这条路被审查掐死——合并平均等价 g=0.5 mixing，把 §3.2 的衰减重新引进来。改为 LRU 整体丢弃：丢"对决策最不重要"的整槽（slot bank 用 [C12] CSM 重要度排序，grid 用 timestamp eviction）。代价是被丢的内容**彻底不可恢复**——但这条代价是诚实的（信息要么保留，要么丢失，不存在"温和淡化"那种第三态）。

**LRU 丢弃后 slot 的状态必须显式说清**——这是审查里被点到的工程接口（独立技术审查 Issue 1）。slot bank 的 key 是 episode 起点 frozen 的随机方向；丢弃后**单纯把 v_i 置零**会让 cos(q, 0) 退化为 softmax 的奇点（数值上未定义）；**重采样 k_i** 又会让新 key 与 bank 内已存在的 frozen key 在 d_s=256、K_s=64 下期望最大 cosine ~0.15-0.2、引起后续路由污染。倾向方案是：**保留 k_i 不变 + v_i 置零 + 维护一个外部布尔 mask `slot_free[i] ∈ {0,1}`，softmax 路由时把 free slot 的 logit 减去一个大常数**——这样 free slot 不会被路由到、首次写入时 mask 自动翻成 0、整槽行为干净。这个细节在源架构里没被点透、必须在实现阶段补上 unit test 验证。

**这一节的不易收敛点。**

- **K_s = 64 是不是够。** 200 步任务的 event 数 3-5 远小于 64，LRU 几乎不触发；但 500 步厨房任务可能 80-100 events，触发频繁丢弃。**倾向 K_s = 64 作为 200-步基线、跨域时拉到 128**——这是 §H Trade-off 4 显式承认的代价。
- **LRU 用什么排序。** slot bank 用 CSM 重要度（"删了之后决策变化大的不能丢"），grid 用 timestamp（"长期没读过的可以丢"）——两套判据共用"LRU"名义但语义不同，注意命名一致性。**倾向：分槽分别按各自判据**，不强求统一规则。
- **跨 episode 是否清空。** 当前架构 episode 间清空 M_sem（放弃跨 episode 学习）——这是 §H Trade-off 2 显式 disclaim 的边界。**倾向先严格 episode 边界**，跨 episode persistent slot 作未来扩展（需要重新设计 hash collision 处理）。

---

## § 3. 为什么 γ 必须是连续标量、不是 0/1 分类

切了段就要写入。但"什么时候触发写入"这件事，在 problem statement 里被 HC-5 标过：**单一边界分类器是单点故障**——一个 learned subtask classifier 训坏了，整个写入触发器报废，下游全垃圾。Mem-0 那一系的 GT vs learned 性能差（28.5% → 45.3% on long horizon）就是这条的实证。

直觉上的两条修法都不行：

- **多个 classifier ensemble 投票。** 治标——任何一个 classifier 都用同样的"这帧是边界"二分类目标在训，集体翻车的概率比单个翻车不会低多少。HC-5 的本质是"避免依赖单一组件类型"，不是"多放几个同类型组件"。
- **拉高 classifier 的 confidence 门槛。** 等价于把分类器输出二值化得更狠——边界处的不确定性进一步丢失，反而更糟。

**正确的方向是把"是不是边界"从 0/1 分类换成 [0,1] 连续标量。** 边界本来就是带不确定性的潜变量后验（这一点跟 memory-vla 那边软边界 MLP 的论证同源）。连续标量的好处是：

- 写入幅度可被软门控调节，γ=0.7 与 γ=0.85 的差别在下游可微体现，不需要在阈值附近反复跳跃。
- 即使 ESPC 在某帧给出"模糊"输出（γ ≈ 0.5），下游也是按比例写入而不是要么写要么不写——错误被衰减而不是放大。
- 留下了 hindsight 监督的接口：γ̂ ∈ [0,1] 当 BCE target 自然，而 0/1 标签会跟回归 / sigmoid 损失打架。

但连续标量带来的问题是：**ψ 自己怎么训**。这分两块讲——主输入是什么、监督来自哪。

### ESPC 的输入与计算路径

主输入是 **M_work + h_t**——ψ 看 M_work 全部 K_w·N=2048 token 预测下一帧 hidden state ĥ_t|t-1，再算 h_t 与 ĥ 的预测误差。预测误差按两条投影分通道：

- **e_geo_t**：token 局部空间投影差（d_proj=64），捕捉物体位置 / 手部动作的物理变化
- **e_sem_t**：全局语义投影 cosine 差（pool 后），捕捉子任务进度突变

经 EMA 标准化（μ_*、σ_* 用全数据 EMA 维护、frozen、非可学）后过 sigmoid 得 γ_geo / γ_sem。**EMA 系数固定（0.99）、参数不可学**——任何让标准化层可学的设计都给 L_main 留了一条经标准化层污染 ψ 的旁路。

### 监督结构——把"训练时"和"部署时"分开

这件事对 CHIME 架构是 **train-deploy 不对称性的根源**，不写明会让所有下游误以为 γ 推理时也能拿 hindsight。

- **训练时——L_HCS 主监督**。[C10] HCS-H 算 ∂a*_{t+Δ}/∂o_t Frobenius 范数 + RUDDER LSTM g_θ 一阶差分作辅助，分通道融合 → γ̂_geo / γ̂_sem ∈ [0,1]。L_HCS 是 BCE(γ_*, sg(γ̂_*))，γ̂ 在 target 侧 sg（γ̂ 是 target 不是可学量）。这条 loss 把 ψ 训成"对未来确实有用的帧给出高 γ"。
- **部署时——γ 自跑、L_HCS 退出**。部署时未来不可知，[C10] 不存在；ψ 只用 prediction error + EMA + sigmoid 出 γ。**hindsight 影响沉淀在 ψ 参数里**——这是 train-only 信号在 deploy-time 留下来的唯一形式。

部署时 γ 的质量受两个分布偏移影响：(i) test 视觉分布偏移会让 prediction error 失准（distractor-heavy 场景误触发的根源），(ii) 训练数据未覆盖的子任务结构会让 ψ 没有合理输出。这两条都是 §H Trade-off 1 显式承认的代价——**域内 evaluation 时 γ 表现接近训练分布；跨域 zero-shot 时 γ 退化为低质量代理**。

### 这一节的不易收敛点

- **EMA 系数 0.99 是不是够。** 训练初期 e 的分布在剧烈变化，0.99 跟不上 → γ 长期被压在 0.5 附近无方差（§I.4 红旗 #2 就是这条）。**倾向用 warmup：前期偏小、稳定期回到 0.99**——但具体曲线得看 e 的分布稳定时间，不在设计阶段拍板具体步数。
- **几何 / 语义共享一个 ψ 还是两个。** 当前方案是同一 ψ + 两条独立 projection——参数省、训练简单；fall-back 是 ψ_geo + ψ_sem 两个独立 1-layer Transformer（参数 +10M）。**倾向先共享、监控分通道 γ 是否解耦**——v1 的"分布密度模式不同"实证强度不足以直接上 +10M 参数。
- **sigmoid vs hard threshold + soft margin。** sigmoid 在极端 e 下饱和（γ 长期 ≈ 1 或 0）→ 软门控失效；hard threshold + soft margin 保留极端区分度但不平滑。**倾向 sigmoid + 监控饱和率**——饱和率持续偏离正常区间就切 hard threshold + soft margin（§I.4 红旗 #2 的对策）。
- **EMA 标准化层的可学性**（独立技术审查 Issue 8）。EMA μ_*、σ_* 已声明 frozen 不可学；但**生成 e_geo / e_sem 的 geo_proj / sem_proj 必须显式 frozen 或仅 L_HCS 可训**——若让 L_main 经预测误差路径反传到 projection，HC-3 隔离形同虚设。这条不在设计层面，在 sg 矩阵层面（见 §5.4 扩展表）。

---

## § 4. 为什么必须双通道，又为什么是按精度分而不是按位置分

切好段、有了 γ，下一步是写到哪。先说为什么不能单 bank：

**HC-7 的硬数字**。几何信息要表达"红块在 (0.5, 0.5, 0.3)"——亚厘米精度，d_g 维度可以低（局部几何熵低，d_g=64 够）；语义信息要表达"红块入抽屉这件事已发生"——任务进度级，d_s 必须高（事件语义熵高，d_s=256）。共表示在同一 latent 上的失败模式是 §3.4 给的：要么几何被语义平滑（low-d 表征丢精度）、要么语义被高 d 表征污染（事件被几何细节淹没）。

**为什么按精度分而不是按位置分？** v1（CW-Mem）的方案是 VLM 端 + action expert 端的双层放置——按"在 forward 路径的哪个位置"分。审查认为这条路的依据是间接的：HC-7 的本质约束是"信息精度"，而 forward 位置只是精度的一个相关代理。直接按精度分（geometric 通道 vs semantic 通道）跟 HC-7 的字面意思对齐，**且写入触发器（γ_geo vs γ_sem）和 aux loss（[C12] 仅作用 [C4]）也是分通道的**——结构性差异给了 stop-grad 隔离一个干净的边界。

但这条 claim 的依据强度只有 **[中]**（§F 选择 2 显式标记）。HC-7 的 "30 vs 10 bits 数量级"证据并不严格支持"两个独立 bank"——同样支持"单 bank 双 head"。维持双通道的理由不在精度本身，而是 **独立写入触发器与独立 aux loss 需要结构性分离**：若共用 bank，L_HCS（作用 ESPC）和 L_CSM（作用 [C4]）在同 bank 上互相污染（因为 [C12] leave-one-out 会动到 [C3] 写入的内容）。

### 高精度物理坐标通道——按 3D 索引

几何通道要装的是"亚厘米精度的物理位置"——索引轴必须是 3D 坐标本身，而不是某种从 hidden state 学出来的向量。原因是空间精度的本质是"同一个位置反复观测自然累加成强信号、不同位置自然分离"，这个性质只有在按物理坐标索引时才免费成立。具体落地是按 NICE-SLAM 风格的多分辨率 voxel grid，下文写作 **M_geo**：

```
M_geo_coarse ∈ R^{8×8×8×64}     场景级布局
M_geo_mid    ∈ R^{16×16×16×64}  物体级位置
M_geo_fine   ∈ R^{32×32×32×64}  亚厘米精度
```

写入流程是每帧每个 token 通过 token_to_voxel MLP 投到 3D 位置，对周围 8 个邻居做三线性插值加权 scatter。**关键不变量**：每帧最多动 ~2000 voxel（256 token × 8 邻居），3 万多 voxel **字面意义上一动不动**。这是 delta-rule 零衰减的代数基础——"加 0 = 不变"是字面意思。

容量满后用 timestamp eviction（[C8] 读时盖 timestamp，最久没读的 voxel 被丢）。**注意**：grid 的 LRU 跟 slot bank 的 LRU 不是同一套算法（grid 按 timestamp，slot bank 按 [C12] CSM 重要度）——共用 "LRU" 名义但判据不同，文档里 grid 那条改名 timestamp eviction 以避免混淆。

### 事件级内容寻址通道——按内容索引

语义通道装的是"红块入抽屉这件事已发生"——索引轴必须是内容相似度，而不是物理位置（事件没有 3D 坐标）。具体落地是 **M_sem**：64 个 slot，每个 slot 由 (k_i ∈ R^{256}, v_i ∈ R^{256}) 组成。**关键反直觉点**：k_i 是 episode 开始时随机初始化 + frozen 的随机方向——**不携带任何预设语义**。slot 装什么由 first-write 事件随机绑定（q_t 与 64 个 frozen k 算 cosine，softmax 后最相似的 1-2 个 slot 拿走 ~0.7 权重 + 加性写入）。

为什么 frozen 而不是可学？可学 key 会被 L_main 训成"对决策有用"的方向漂移——t=10 写入 slot #5 时 k_5 是某个方向、t=80 想读 slot #5 时 k_5 已经飘到另一个方向，**两次访问的"地址"指向了不同的语义内容**，记忆错位。frozen 随机 key 等价于哈希表的桶号——地址本身不携带语义，**语义在 value 里、由内容决定**。这是规避 HC-3 的关键（任何与路由 / mixing 相关的可学参数都会被 BPTT 训坏）。

emergent 语义机制：相似事件的 q 仍与 k_5 余弦最相似 → 继续路由到同一 slot 累加强化；不相似事件路由到不同 slot 自然隔离——**等价于可微哈希表**，soft 加权而非 hard top-1 保 HC-2 全可微。

### 这一节的不易收敛点

- **几何通道的 token_to_voxel MLP 怎么训。** 当前方案靠 BC + L_PRH 间接监督——没有 RGB-D 显式监督，学到的是"任务相关的 3D 抽象投影"而非几何重建。**倾向先这样**，但若 distractor-heavy 场景下 grid 被无关几何写满（§H.5 失败模式 1），需要回退到加 RGB-D 自监督 L_GC。
- **slot 数 K_s 与 hash collision。** K_s = 64 在 200 步任务上 3-5 个 event 远低于上限，collision 概率低；但跨域到长任务（500 步、80-100 events）时 collision 显著——两个不同事件被 q 投到同一 slot，加性混合后 read 时变模糊。**倾向 K_s = 64 + L_CSM 强制异质化**，跨域时拉到 128。
- **写入位置由 token_to_voxel 决定，会不会把背景 token 也写满 grid。**（独立技术审查 Issue 6——这条原本被写成 emergent，是过强的 claim）严格地讲，L_PRH 不直接惩罚"背景 token 占用 grid"——它只奖励"m_t 能预测未来"。背景 token 被压缩到无关位置确实可能 emergent 出现（任务无关 token 写哪里都不影响未来预测、参数更新就在零空间里飘），但**这件事没有结构性保证**——256 token × 8 邻居 × 100 帧 = 上界 200k 写入对 ~36k voxel 容量，不靠 γ_geo 软门控压幅度 + 帧级 sg 调度的话很容易 4-5 帧就饱和。所以**倾向监控 grid 占用率**——占用率持续偏离 sparse 范围就触发对策，对策有两条：(a) 加 distractor 增强训练数据（数据侧）、(b) 加 token 级 L1 sparsity 项到 grid 占用上（结构侧）。当前架构选 (a) 因为简单，但 (b) 是兜底——把"背景 token → 边界"从 emergent claim 降级为"emergent + 监控 + 兜底正则"三层防御。

---

## § 5. 整套方案的端到端 workflow

读到这里，每个模块的 why 都已经在前几节给了：delta-rule 解决 §3.2 衰减，连续 γ 解决 HC-5 单点，双通道解决 HC-7 共表示，sg(γ) 解决 HC-3 隔离。但这些论证是分散的，读者头脑里还没有一张完整的图。这一节就是这张图——**跟着抽屉任务（"开抽屉、放红块、关抽屉" T=100 步）从进 forward 到出动作走一遍**，每碰到一个模块就把它的具体机制就地展开。

### 5.1 数据进出的边界条件

**输入侧**。每步 t 机器人接到 RGB + proprio (o_t)。**训练时**额外有离线已知的 trajectory 全长 (o_{1:T}, a*_{1:T})——hindsight pipeline 用这个反推 γ̂。**推理时**未来不可见，仅 (o_{1:t}) 可见，γ̂ 不可得。

**输出侧**。[C9] action expert 吃 c_t 经 flow matching ODE 出 a_t ∈ R^7 下发执行器。

**主干前段（训推共享）**。o_t 进 [C1] SigLIP-ViT-L (frozen + LoRA) 出 h_t ∈ R^{N×d_h}（N=256, d_h=1152）。这一段从 baseline VLA 继承、本方案不动。**所有改动从 h_t 之后开始**。

### 5.2 训练时跟着 t=20 → t=60 → t=90 走一遍

```
o_t (RGB + propio)
       │
       ▼
[C1] VLM 主干 (frozen + LoRA)  ────►  h_t ∈ R^{256×1152}
       │
       ├──► [C2] FIFO M_work (last K_w=8 frames)
       │
       ├──► [C5] ESPC ──► γ_geo_t, γ_sem_t ∈ [0,1]
       │           │  主输入: M_work + h_t
       │           │  ψ 预测 ĥ_t|t-1 → e_geo / e_sem → EMA → sigmoid
       │           │  训练时 L_HCS BCE 推 γ → γ̂ (来自 [C10])
       │           ▼ sg(γ_*)  ←──── 写入瞬间 sg, 阻断 L_main → ψ
       │
       ├──► [C3] Geo 写头 → [C6] M_geo grid (3D 物理索引)
       │           │  delta-rule: M_geo[8 邻居] += sg(γ_geo)·α_ℓ·w_j·φ_ℓ(h_t)
       │           │  容量满 → timestamp eviction
       │
       ├──► [C4] Sem 写头 → [C7] M_sem slot bank (content key)
       │           │  q = MLP_q(pool(h_t)), softmax(cos(q, frozen k_i)/τ)
       │           │  delta-rule: v_i += sg(γ_sem)·w_i·φ(q)⊗v
       │           │  容量满 → CSM-importance LRU 丢弃
       │
       ▼
[C8] 读出: cross-attn(N_q query, M_work + M_sem) + spatial sample(M_geo)
       │   →  c_t ∈ R^{(N_q + K_w)×d_h}
       ▼
[C9] Action Expert (flow matching + LoRA) ──► a_t
```

**(t=20，open 阶段中段)**。手伸向把手，几何在加速、子任务进度无突变。
- ψ 在 M_work[13:20] 上预测 ĥ_20|19，e_geo_20 中等（手在加速）→ γ_geo_20 ≈ 0.6；e_sem_20 低 → γ_sem_20 ≈ 0.15
- [C3] 中等强度写入手部 + 把手附近 voxel：fine voxel (≈ 把手位置) 累积 0.056 量级 / 帧
- [C4] 几乎不写入：sg(γ_sem)=0.15 × 任意 w_i 都很小
- 训练时 L_HCS：[C10] 算 ∂a*_36/∂o_20（Δ=16）→ γ̂_geo_20 ≈ 0.7（"未来 16 步抓把手依赖此帧观测"）→ BCE 推 γ_geo 上升

**(t=60，place 阶段后段)**。红块刚放下，子任务进度突变。
- e_sem_60 飙高 → γ_sem_60 ≈ 0.85；e_geo_60 中等 → γ_geo_60 ≈ 0.7
- [C4] **强写入**：q_60 与 64 个 frozen k 算 cosine，假设 k_18 最相似（纯偶然），w_18 ≈ 0.7。v_18 += 0.85·0.7·φ(q_60)⊗v_60 ≈ 量级 0.6 → **slot #18 从此装上"红块入抽屉"事件语义**
- [C3] 同时写入红块当前位置（抽屉内坐标）的 fine voxel
- 训练时 L_HCS：[C10] 算 ∂a*_76/∂o_60 → γ̂_sem_60 ≈ 0.85（"未来关抽屉的动作高度依赖此帧的红块入位事件"）

**(t=90，close 阶段后段，红块完全遮挡)**。
- ψ 没有显著 prediction error，γ_sem_90 ≈ 0.4，[C4] 微弱写入 slot #27（"关闭进行中"）
- [C8] 读出：cross-attn 把 q_90 与 64 个 slot key 算 cosine——q_90 与 k_18（"红块入抽屉"绑定方向）仍高余弦相似 → cross-attn 给 slot #18 高权重 → c_90 含"红块在抽屉里"的事件语义
- [C9] 决策"是否重开看红块？"：c_90 含 slot #18 → action expert 知道"那位置有东西" → **决策：继续关到底，不重开**——belief state 跨 30 步保留的具体实现，**纯靠 delta-rule 加性 + frozen key 的内容路由 + cross-attn 读出，不依赖任何长程 BPTT**

### 5.3 训练时多了什么（部署时不跑）

前面 5.2 是部署 forward 路径。训练时还多 4 个组件给写入端反传压力（HC-4 aux loss 必要性）：

- **[C10] HCS-H**：在离线 trajectory 上算 ∂a*_{t+Δ}/∂o_t，Δ ∈ {4, 16}（显存允许时加 64）+ RUDDER LSTM g_θ 一阶差分。两条信号融合 + z-score + sigmoid → γ̂_geo / γ̂_sem。L_HCS 把 γ → γ̂。**这是 §3.1（在线无法预知未来）的唯一突破口——离线时未来已知，反推 target 后 baked 进 ψ**。
- **[C11] PRH**：从 m_t 预测 (o_{t+k}, a*_{t+k})，k ∈ {4, 16, 64}。L_PRH 强制 memory 内容"对未来可预测"——这是 §3.1 的另一条代理（承认无法知道"未来需要"，但要求"未来可由 memory 预测"）。MERLIN 风格。
- **[C12] CSM**：每 mini-batch 抽 4 slot，做 leave-one-out（把 v_i 置零）算 KL(π(a|m_t) ‖ π(a|m_t \ slot_i))，rich 的 slot KL 大、垃圾 slot KL 小。L_CSM = -Var_i(w_i) - β·log Mean_i(w_i)——前项强制异质化，后项强制利用率。回应 §3.4 + 顺便给 [C7] LRU 提供重要度排序。
- **[C13] 反向 Jacobian**：[C10] 同时算 ∂a*_{t-Δ}/∂o_t 处理反向因果场景（柔性接触），罕见但兜底。MVP 默认砍。

### 5.4 stop-grad 矩阵——架构正确性的关键

任何 sg 不一致让 HC-3 隔离失效，整个架构退化。下面这张表是必须显式维护的——独立技术审查 Issue 8 指出原 4 处不够，扩展到 7 处：

| 位置 | sg 谁 | 理由 |
|---|---|---|
| [C3][C4] 写入瞬间 | sg(γ_*) | L_main 不能流回 [C5] ψ 训写入门——HC-3 |
| [C8] L_PRH 路径上 query 投影矩阵 | sg(query) | L_PRH 不能经 query → h_t 污染 [C1] LoRA |
| [C10] 输出 γ̂ 喂 L_HCS 之前 | sg(γ̂) | γ̂ 是 BCE target 不是可学量 |
| [C12] 经 frozen [C9] | 自然 sg | L_CSM 不会反向训 [C9]（否则 [C9] 学到"对 slot 缺失敏感"以 hack L_CSM） |
| ψ 看到的 M_work 内容 | sg(M_work) on L_HCS path | L_HCS 不能经 ψ 输入反传到 M_work 内的 h_{t-7..t-1} → [C1] LoRA |
| [C5] 的 geo_proj / sem_proj | 仅 L_HCS 可训、不接 L_main | 否则 L_main 经预测误差路径污染 ψ |
| [C8] cross-attn 读端被 L_main 训 | 不能 sg（结构性） | 但这条是 §6 提到的"被训成短视"风险——L_aux entropy 正则**不是真正的解、是工程兜底** |

最末两条说明 sg 矩阵不是封闭的——cross-attn 读端被 L_main 训成"只看最近 1-2 帧"是结构性风险，没有干净的 sg 修法、只能靠 L_aux entropy 正则按住。**代码层面这张表必须有 unit test 验证**——任何重构后都要重新审查 sg 在不在正确位置；审查里 Bug 4（γ 在中间层而非最外层 + sg）就是这种事故的具体形式。

部署时三个 train-only 组件（[C10][C11][C12][C13]）完全不跑——它们的影响已沉淀在 [C5] ψ 与 [C3][C4] 写头的参数里。

### 5.5 训练 vs 部署的本质区别——三处显式不对称

- **触发器**：训练时 γ 由 ψ 自跑 + L_HCS BCE 推向 γ̂；部署时 γ 完全由 ψ 自跑、γ̂ 不存在。
- **数据流**：训练时跑 forward + 4 个 train-only 组件 + 5 个 loss；部署时只跑 forward 路径。
- **Bank 内容**：训练时 bank 装的是"被 hindsight 校准过的 γ 写入的 entry"；部署时装的是"仅 prediction error 校准的 γ 写入的 entry"——分布偏移由 §H Trade-off 1 显式承认。

---

## § 6. 读出：为什么三路 KV 拼接、且必须分采样方式

读出端要满足三个前提才能 work，且**这三条是 CHIME 特有的**（不是从一般 attention 文献抄来的）：

- **frozen-key 路由稳定性。** M_sem 的 slot 通过 frozen 随机 k_i 路由——这要求 query 投影 q_t 必须落在与 k_i 同一个 d_s=256 空间，否则 cosine 相似度退化为噪声。
- **三路 KV 长度差三个数量级。** M_geo voxel 数 ~4.2 万，M_work + M_sem 才 ~2k——直接拼成同一组 KV 喂 cross-attn 推理时间被 M_geo 拉爆（HC-6 直接破）。
- **delta-rule 累加幅度跨范围大。** 同位置反复观测累加到 ~3.0、单帧写入只有 ~0.05——读端 attention 不能被高幅度 voxel 主导导致低幅度 slot 被淹没。

第二条和第三条共同把"单一 cross-attn over 全部 KV"这条路堵死。三路 KV 长度对照：

- M_work：K_w·N = 8 × 256 = 2048 token——可控
- M_sem：K_s = 64 slot——可控
- M_geo：8³ + 16³ + 32³ ≈ 4.2 万 voxel——**遍历就 OOM**

逼出**分采样方式**：

- **M_work + M_sem**：N_q = 16 个 learnable query 做 cross-attn，KV 总数 = 2048 + 64 ≈ 2k，可控
- **M_geo**：N_geo_q = 16 个 spatial query 做**三线性采样**——每个 query 取 8 个邻居 voxel，coarse 层为主（8³=512 voxel）、mid/fine 层只在 coarse 命中后局部精化，总采样 ~128 voxel 而非 5e4——这是 HC-6 实时性的关键

两路读出拼接成 c_t ∈ R^{(N_q + K_w)×d_h}，喂 [C9]。

### 读端的隐性失败模式：FIFO 被训成短视

[C2] M_work 本身没有可学参数（FIFO 就是 ring buffer），但**读端 cross-attn 是可学的**——L_main 可以训出"只 attend M_work 最后 1-2 帧"的短视读出，等价于 §2 提到的"门控更新短视"被推到 read 端。这条审查 Bug 7 的对策是 **L_aux attention entropy 正则**：

$$\mathcal{L}_{\text{aux}} = -\lambda_{\text{ent}} \cdot \mathbb{E}_t [H(\text{attn weights of [C8] over } M_{\text{work}})]$$

λ_ent = 0.01，强迫 attention 不能完全集中在最近 1-2 帧。**这条不是真正的解，是最低限度的工程兜底**——没有它的情况下短视退化是免费的，loss 看不出来。

### 这一节的不易收敛点

- **N_q 与 N_geo_q 取 16 是不是够。** 太小（N_q=4）信息瓶颈太窄，长程能力受限；太大（N_q=64）推理延迟逼近 HC-6 上限。**倾向 N_q = N_geo_q = 16 起步，根据 [C8] cross-attn 的 attention entropy 分布调**——entropy 持续接近 0（attention sink）说明 query 太少；entropy 接近最大（均匀化）说明 query 太多。
- **三路读出的相对权重谁定。** 当前方案是直接拼接喂 [C9]，权重让下游 attention 自己学——不显式 gate。**倾向不加 gate**——gate 容易学到把某一路关到 0（与 memory-vla 那边双通道融合"separate heads 而非 gate"同源）。失败模式 head-collapse 靠 §0 监控痕迹（per-head L2 比 + DiT 下游梯度比）抓。
- **M_geo 的三线性采样在 fine 层会不会 cache miss。** 三线性插值是局部操作，但 fine 层 32³ = 32768 voxel 对 GPU memory access pattern 不友好。**倾向先 profile 再决定**——若 cache miss 严重就把 fine 层降到 16³（与 §0.7.4 MVP 简化一致）。

---

## § 7. 训练：多 loss 联合 + 隐式课程稳住整个架构

到这里搭出了完整结构：[C1] → [C2/C5] → [C3/C4] → [C6/C7] → [C8] → [C9]，加 4 个 train-only 组件给写入端反传压力。下面把**为什么需要这么多 loss** 的因果链先讲一遍——表只是 bookkeeping，逻辑链才是论证。

**L_main 不能独自支撑写入端**（HC-3 + HC-2 联合判死）——L_main 经 cross-attn 反传到 [C3][C4] 写头，但写入瞬间 sg(γ) 让 ψ 拿不到 L_main 梯度；同时 BPTT 跨百步在显存上不可达。**逼出 L_HCS**——把 hindsight saliency 当 BCE target 给 ψ 一条直接梯度路径（绕过 BPTT、绕过 sg）。但 L_HCS 只训 ψ 的"什么时候写"，**不**保证写进去的内容质量——逼出 **L_PRH**：要求 m_t 对未来可预测，给 [C3][C4] 写头 + [C6][C7] memory 内容一条独立监督。L_PRH 仍然不约束 slot **之间**的差异——slot 可能塌成全装相似中等内容（hash collision），逼出 **L_CSM**：leave-one-out 强制 slot 异质化。最后 [C8] cross-attn 读端被 L_main 训成短视的风险（§6 末尾），逼出 **L_aux**：attention entropy 正则按住短视退化。**这五个 loss 不是"堆 trick"，是 sg 矩阵把 L_main 的反传路径切成五段后必须各自补回的最低限度反传压力**。

可选第六个 L_GC（geometric consistency，仅 RGB-D 数据下启用）作为 M_geo 的额外正则，MVP 默认 disabled。

$$\mathcal{L} = \mathcal{L}_{\text{main}} + \lambda_1 \mathcal{L}_{\text{HCS}} + \lambda_2 \mathcal{L}_{\text{PRH}} + \lambda_3 \mathcal{L}_{\text{CSM}} + \lambda_4 \mathcal{L}_{\text{GC}} + \mathcal{L}_{\text{aux}}$$

λ_1 = 0.3, λ_2 = 0.5, λ_3 = 0.1, λ_4 = 0.2（标定值，需 sweep；λ_4 仅在 RGB-D 数据上启用，否则项整体置零）。

| Loss | 作用对象 | 解决约束 | 删了会怎样 |
|---|---|---|---|
| L_main (BC flow matching) | [C1] LoRA + [C8] + [C9] LoRA + [C3][C4] 写头 | 基础 | 不可删 |
| L_HCS (BCE on γ) | [C5] ψ + projections | §3.1 + HC-2 + HC-5 | γ 退化为纯 self-supervised；长程 SR -25% |
| L_PRH (predict future obs/act) | [C3][C4] 写头 + [C6][C7] memory + [C11] heads | HC-4 + §3.1 代理 | 长程 SR -15%（短程无差） |
| L_CSM (slot heterogeneity) | [C4] MLP_q/MLP_v | HC-4 + §3.4 | slot collapse；长程 SR -10% |
| L_GC (geometric consistency, optional) | [C3][C6] | M_geo 一致性 | RGB-D 数据下 grid 漂；非 RGB-D 数据上不启用 |
| L_aux (attention entropy) | [C8] cross-attn | Bug 7 对策 | 短视读出风险 |

### 5 个 loss 同时跑没有 cold-start 塌？——必须诚实承认隐式课程

原草案声称"跟 memory-vla 那边'两阶段课程'不一样、CHIME 这边 5 个 loss 一起跑没有冷启动硬约束"——独立技术审查 Issue 5 把这条 claim 戳破：**CHIME 实际上有隐式课程，只是没在文档里明说**。

具体动力学是：训练 step 0-10k γ ≈ 0.5 ± 0.05（EMA 还没收敛、ψ 还没被 L_HCS 推动），这时 [C6][C7] 接收的是"均匀中等幅度"的写入——背景和事件不分轻重一起写进去、慢慢饱和成噪声；而 L_PRH 此时读到的就是这种噪声 memory，反向梯度把 [C3][C4] 写头训到一个塌掉的局部最优。**§8 第一处痕迹（"γ_sem 塌缩到 0.5 ± 0.05"）写明了是红旗——这本身就是冷启动模式的具体形式**。

正确的描述：CHIME 不是"无课程"，是**把课程藏在三个 schedule 里**：

- **EMA warmup**：前期 EMA 系数偏小（让 e 标准化跟得上分布漂移），稳定期回到 0.99。
- **L_HCS 渐进接通**：M1（E1 通过前）L_HCS 不接通（[C10] 还没 calibrate）；E1 通过后才把 λ_1 从 0 anneal 到 0.3。这件事在源架构 §I.3 milestone 上是隐含的，但从来没作为"课程 schedule"显式承认过。
- **写头早期 frozen**：可选——若 §8 监控显示前 1-2k step ‖γ_*‖ 方差崩塌，临时把 [C3][C4] frozen 直到 γ 解出方差。这是反应式对策、不是预设课程。

**所以 §7 的诚实表述是**：CHIME 多 loss 联合训练、不需要 memory-vla 那种显式 stage A → BC 切换，但**有隐式课程**（EMA warmup + λ_1 渐进接通 + 反应式 freeze）。MVP 配置（L_HCS 整个砍掉）则等价于把课程退化到只剩 EMA warmup 一条——更稳但放弃 §3.1 后视回应。

### Stop-grad 矩阵的工程意义

5 个 loss 同时跑的稳定性**完全依赖 stop-grad 矩阵的正确性**——任何一处 sg 漏写，架构退化为某个简化版本：

- 漏 sg(γ_*) 在 [C3][C4] 写入瞬间 → L_main 流回 [C5] ψ → BPTT 把 γ 训成全 1 → 短视写入（HC-3 失效）
- 漏 sg(query) 在 [C8] L_PRH 路径 → L_PRH 经 query → h_t 反传 → 污染 [C1] LoRA → ψ 输入分布漂 → ESPC 失效
- 漏 sg(γ̂) 在 [C10] 喂 L_HCS 之前 → L_HCS 反传到 [C10] → γ̂ 被训成"让 BCE 容易最小化"的 target，hindsight 信号失真
- 漏 frozen on [C9] in [C12] → L_CSM 反传到 [C9] → [C9] 学到"对 slot 缺失敏感"以 hack 这个 loss

**这件事在审查中反复栽跟头**（§J 列了多处类似 bug）。文档 §B 的数据流图必须显式标注每处 sg，代码必须有 unit test 验证 sg 在正确位置——这不是 nice-to-have、是架构成立的前提。

### Hindsight pipeline 的 base policy 选择——chicken-and-egg

[C10] 算 Jacobian 需要一个 frozen base policy 当"标尺"。这个 base 选什么直接影响 J 的语义：

- **π0.5（无记忆预训练）**：立即可用，无 chicken-and-egg；但 J 反映"无记忆策略对当前帧依赖"——是任务相关帧的 lower bound、信号偏弱。
- **CHIME early checkpoint**：J 信号更锐、反映长程信息需求；但 chicken-and-egg：早期 CHIME 还没学会用记忆，J 有偏。
- **π0.5 fine-tuned on target**：中间方案、信号较准；但工作量加倍。

**倾向**：E1（M1, week 4）第一轮用 π0.5 起步，IoU > 0.4 通过后 E2/E3 切到 CHIME early checkpoint。这是个 sequential dependency：base 不准 → γ̂ 不准 → ψ 不准 → CHIME checkpoint 不准——所以 E1 通过率是整个项目最大赌注（§F 选择 3 标 [弱]）。

**但这条路径有一个独立技术审查（Issue 4）点出的 circular bias 风险**——比 chicken-and-egg 更严重：用 π0.5（无记忆）算 ∂a*_{t+Δ}/∂o_t 的字面意思是"未来 Δ 步的**无记忆策略**最优动作对当前帧观测的敏感度"。但 CHIME 真正想找的"值得记的帧"是**只对带记忆策略才显著的帧**——比如 t=60 的"红块入抽屉"事件，它的价值在 t=90 决定是否重开抽屉时才兑现，而无记忆 π0.5 在 t=90 根本用不上 t=60 的信息（它没记忆），所以 ∂a*_90/∂o_60 在 π0.5 上反而是**小**的。换言之 π0.5 base 系统性低估了 CHIME 真正要的那种长程记忆相关帧。

切换到 CHIME early checkpoint 不是 chicken-and-egg、是 **circularity**——checkpoint 的 J 反映"CHIME 已经学会注意什么"，再训 ψ 就是"用学到的注意力模式自己强化自己"。**正确的修法是引入独立的 ground-truth saliency 校准**：在小验证子集上用 oracle counterfactual（drop frame, replan with full demo trajectory）或高容量 offline RL value function 拟合 demos 算出的 saliency 当 IoU 比对的真值——而不是直接拿人工标注的子任务边界比对。**这条修正还没完全落到 §I.0 数据准备里**——是必须在 E1 跑之前补的工作量（设计阶段倾向，但具体 oracle counterfactual 算法仍然开放）。

如果不修而直接跑 IoU @ 0.3 ≥ 0.4 against 人工边界，**E1 通过本身可能不能 validate hindsight 信号质量**——只能 validate "Jacobian 找到的高敏感帧大致对齐人工标注的子任务过渡"，而这两件事不严格等价。**这是 §11 单点故障"HCS Jacobian 信噪比是项目存亡判决"在更细一层的具体形式**。

### 这一节的不易收敛点

- **5 个 loss 之间 conflict 的处理。** λ 的 sweep 空间是 5 维的——人工调几乎不可能稳定。**倾向用 GradNorm / PCGrad 自动平衡**（§I.4 红旗 #6 的对策）——但若 multi-loss balancing 第 8 周仍不稳，整个项目要重新评估。
- **Δ ∈ {4, 16, 64} 在 batch=32 上能不能塞下。** Δ=64 时 frozen base policy 跨 64 步 activation ≈ 30 GB at fp16，接近 80GB 上限——**OOM 风险 ~30%**。fall-back 是 Δ ∈ {4, 16} + batch=16，但这意味着任务真实因果跨度 > 64 步时 L_HCS 无法直接监督（§H Trade-off 7）。
- **L_PRH 在 k=64 上下不下降。** 红旗 #3 显式列：若 L_PRH 在 k=64 不降只在 k=4 降，说明 memory 实际只缓存 ~10 步，长程能力为零，§3.2 claim 实证伪。**这条是 M3 的硬门槛**——不通过整个 §3.2 主张推翻。

---

## § 8. 监控：Loss 是滞后指标

跟 memory-vla 那边同源的论证：当一个分支静默退化时，loss 通常会跟"那个分支被关掉的 baseline"一样平稳下降——只是天花板低。CHIME 的失败模式比 memory-vla 多一类：**stop-grad 漏写让 sg 隔离失效**。这类失败 loss 完全看不出来，必须直接观察必然留下痕迹的中间量。

下面六处痕迹不是"指标罗列"——每一个都是某个失败模式在物理上的必然反映：

**第一处痕迹：γ 是否塌缩。** ψ 训坏的两种典型：(a) γ_sem 长期 ≈ 0.5 ± 0.05（无方差）—— EMA 标准化吃掉所有信号、写入触发器失效（红旗 #2）；(b) γ_sem 长期 ≈ 1（饱和写入）—— BPTT 截断 + sg 漏写，HC-3 失效。监控 γ 的 running median 和方差分布。

**第二处痕迹：Slot 利用率。** 健康的 slot bank 在 200 步任务上应该有 3-5 个非零 slot，其余 ≈ 0。退化模式：(a) 1-2 slot 占 80%+ 写入（hash collision、L_CSM 没生效，红旗 #4）；(b) 所有 slot 装相似中等内容（L_CSM 项 A 没起作用——异质化失败）。监控 ‖v_i‖ 分布的方差和 top-1 占比。

**第三处痕迹：L_PRH 在不同 k 上的下降。** k=4 降但 k=64 不降 → memory 只缓存短期，长程能力为零（红旗 #3）。三个 horizon 必须分别追踪——单一总 L_PRH 看不出来。

**第四处痕迹：Per-head 输出 L2 比 + DiT 下游梯度比（双通道是否还活着）。** 三路读出（M_work / M_sem / M_geo）走同一个 [C8] cross-attn 但分别经独立 query——监控每路输出的 L2 范数比 + [C9] 对每路的梯度幅值比。任一路持续偏向 0 = 静默 collapse。

**第五处痕迹：Attention entropy 分布。** [C8] 的 attention 应该既不均匀（无差别）也不集中（attention sink）。两端都是退化：均匀化 = key 区分度消失（M_sem 几乎全空 / 都装相似内容）；sink 化 = 某个 slot logit 系统性偏高、其它压死。

**第六处痕迹：Wall-clock per step 分布。** 5 个 loss + Δ=64 frozen base policy 跨步 forward 在 batch=32 上估计 800-1000 ms/step（H100 BF16），A800 上 1.4-1.8 s/step。监控 P50 / P95——超过预算就触发 fall-back（Δ ∈ {4, 16} 或 batch=16）。

每 1k step 看一次曲线。这六个 scalar 不是"有就最好"——**没有它们就没办法判断 5-loss + 11-component 架构到底有没有 work**。

---

## § 9. 未确定项总览：什么是"不易收敛"

研究构想阶段大量未定项是正常的。"未定"有四种来源，每种对项目意味着完全不同的事情。

### 9.1 接口悬空类——一个未定决定上下游多个论证的成立条件

**最尖锐的一处是 [C10] HCS-H 的 base policy chicken-and-egg**（§7 已述）。base 不准 → γ̂ 不准 → ψ 不准 → CHIME checkpoint 不准。这是 sequential dependency，不能靠 ablation 闭环——E1 不通过整个 §F 选择 3 必须 fallback 到 MVP（disable [C10][C12][C13]）。**这件事的 IoU @ 0.3 ≥ 0.4 阈值是项目存亡判决点**。

类似的还有 stop-grad 矩阵——任何一处 sg 漏写整个 HC-3 隔离失效。这不是设计阶段未定，是**实现阶段必须有 unit test 兜住**的工程要求。

### 9.2 跨模块设计选择类——选项明确但交集要靠实验摸

**[C5] ψ 是否分通道**（共享 ψ + 双 projection vs ψ_geo + ψ_sem 两个独立）。前者参数省、后者信号更纯。**倾向先共享、监控分通道 γ 解耦度调**。

**M_geo 的多分辨率分配**（α_coarse, α_mid, α_fine 当前 0.5/0.3/0.2）。先验上 coarse 收敛快、fine 信号锐——但具体比例是实验问题。

**[C12] CSM 抽样数 4 vs 全 64**。全做计算太贵（每 mini-batch 256 次 frozen [C9] forward），抽 4 估计够 + 长期覆盖所有 slot。**倾向抽 4 起步，看 slot 重要度估计的方差**。

### 9.3 实验决定类——没有先验最优值，只能看曲线

K_s slot 数（200 步任务 64 够、500 步任务可能要 128）；λ_1..λ_3 loss 权重；Δ 默认 {4, 16} 还是含 64（显存允许时加，否则 fallback）；EMA warmup 系数曲线；N_q / N_geo_q 各 16 是否需要调；α_a in L_PRH（obs vs action 权重比）。

**早期实验集中在 9.1 类（base policy chicken-and-egg），9.2 类做主线 ablation，9.3 类作为参数 sweep 对象。**

### 9.4 架构哲学类——一旦定了影响整个论证骨架

**按精度分通道 vs 按位置分（v1 vs v2）**。v2 已倾向按精度分（geometric grid + semantic slot），但 §F 选择 2 标 [中]——单 bank 双 head 仍是合理 fallback，需在控制总容量前提下消融对比。

**Episode 间清空 vs 跨 episode persistent**。当前架构清空（放弃跨 episode 学习），是 §H Trade-off 2 显式 disclaim 的边界。**倾向先严格 episode 边界**——跨 episode 需要重新设计 hash collision 处理 + cross-episode reward attribution。

**三个 train-only aux loss 全保留 vs MVP 砍三个**。MVP（§0.7.4）砍掉 [C10][C12][C13]，γ 退化为纯 prediction error self-supervised + L_PRH——这条 fallback 仍 publishable（"打 HC-3 + §3.2 + §3.4 三条"），但放弃 §3.1 后视回应。**这是项目从 6-month full 退到 3-month MVP 的硬切换**。

### 9.5 快速索引

| 未定项 | 倾向 |
|---|---|
| 写入更新规则 | delta-rule + LRU 丢弃（**已收敛**——§F 选择 1 [中]） |
| 容量管理 | grid 用 timestamp eviction、slot 用 CSM-importance LRU（命名分开） |
| 双通道 vs 单 bank | 双通道（**已收敛但 [中]**——独立写入触发器 + 独立 aux 是关键依据） |
| Bank 维度 | M_geo d_g=64、M_sem d_s=256、K_s=64（200 步基线） |
| 写入门 γ 的形式 | 连续 [0,1] sigmoid 软门控（**已收敛**） |
| ESPC ψ 是否分通道 | 共享 ψ + 双 projection（fall-back: 双 ψ） |
| Hindsight 监督 | Jacobian 主信号 + RUDDER 一阶差分辅助 + 通道分离（**关键赌注 §F 选择 3 [弱]**） |
| Event-segmentation 预测误差作 ESPC 主信号 | **§F 选择 4 [中]**——Baldassano 2017 是 fMRI 观测者实验、与 VLA 控制 backbone hidden state 不同 substrate；在 distractor-heavy 数据上检查 γ 误触发率验证 |
| 反向 Jacobian | **§F 选择 5 [弱]**——MVP 默认砍；纯 design intuition、反向因果 <5% 帧、收益边际 |
| Base policy 选择 | E1 用 π0.5、E2/E3 切 CHIME early checkpoint |
| Δ 上限 | 默认 {4, 16}、显存允许加 64 |
| Aux loss | L_HCS + L_PRH + L_CSM + L_aux（MVP 砍 L_HCS + L_CSM） |
| 读出方式 | M_work + M_sem cross-attn、M_geo 三线性采样、不加 gate |
| Stop-grad 矩阵 | 4 处显式 sg（写入瞬间 / query 投影 / γ̂ target / frozen [C9]） |
| MVP backbone | [C1] = ViT-B（替代 ViT-L）+ [C9] = 1-step flow（消费级硬件可部署） |
| 跨 episode | 严格清空（共享是 v3 扩展） |

---

## § 10. 验证路径：如何知道方案在 work

整套方案的成败压在三件事上：**[C10] HCS Jacobian 信噪比够**（项目存亡）、**stop-grad 矩阵全对**（架构成立）、**delta-rule + LRU 丢弃在长程任务上不塌**（§3.2 claim 实证）。

### M1（week 4）：E1 是项目存亡判决点

在 ~1k BridgeV2 trajectories 上独立运行 [C10]，计算 J^{(16)}_t 归一化结果，与人工标注的子任务边界算 IoU @ 0.3。

| 结果 | 处理 |
|---|---|
| IoU ≥ 0.4 | 通过，继续 full 版本，E2/E3 切 CHIME early checkpoint |
| IoU < 0.4 | **disable [C10][C12][C13] 与 L_HCS、L_CSM**，fallback MVP（§0.7.4） |

**前提工作量**：BridgeV2 没 frame-level event 标注，需要 1-2 人周手标 100-200 traj。这是文档原草案漏掉、审查找补的隐藏工作量。**E1 工程直觉概率 30-50%**——所以整个项目以 fall-back 路径为基准做规划，不以"E1 一定通过"做规划。

### M2（week 8）：[C5] 在 RoboCasa 单子任务收敛 + [C11] PRH 在 1k traj 上独立收敛

[C5] 的独立验证：固定 [C1]，只训 [C5] + L_HCS，验证 γ_sem_t 与 GT 边界 IoU > 0.5。0.5k traj、1 GPU、6 h。

[C11] 的独立验证：固定 [C1][C9]，只训 [C3][C4][C6][C7] memory + [C11] PRH（**无 L_main**），看 m_t 是否真能预测 o_{t+16}。1k traj、1 H100、8 h。**这是个早期 sanity check**——若连这个 baseline 都打不过，说明 memory 写入完全无效。

### M3（week 12）：[C3][C6] + [C4][C7] 联立可训

L_main + L_PRH 联合训练，写头梯度反向通畅、memory 内容非崩塌。**硬门槛重新校准**（独立技术审查 Issue 3）：原版本说"L_PRH @ k=64 显著下降否则 §3.2 推翻"——这条门槛是**两个方向都误校准**的：

- 方向一：L_PRH 是回归 m_t → (o_{t+k}, a*_{t+k})，但 demonstrator 轨迹本身高度自相关，o_{t+64} 经常仅靠 o_t 平滑外推就预测得很好——**通过 L_PRH @ k=64 不 validate memory 真在做长程信用分配**。
- 方向二：Δ ∈ {4, 16} 不含 64 的 OOM fallback 下，L_HCS 直接监督的最长跨度只到 16；声称 L_PRH @ k=64 "carry 长程监督"是把"可预测"和"信用分配"两件事混淆了。

**修法**：M3 硬门槛改为**带控制的双指标**——(a) CHIME L_PRH @ k=64 必须比"无记忆 baseline（同一 L_PRH head 只读 FIFO）"显著低（统计显著性，不是绝对值）；(b) §H Trade-off 7 显式承认"OOM 限 Δ ≤ 16 时，§3.1 hindsight 信号在 >16 步跨度上结构性缺席、L_PRH 仅能验证可预测性、不能验证信用分配"。两者都通过才能说 M3 通过。

### M4（week 16）：RoboCasa 双子任务 + 完整 5-loss 与 baseline ≥ 10% SR 提升

完整架构跑通——这是 publishable claim 的早期验证。

### M5（week 20）：LIBERO-Long + CALVIN ABCD→D 全架构跑通

跨 dataset 验证。CALVIN ABCD→D 的子任务边界标注完整，是 [C5] γ 质量的二次验证。

### M6（week 24）：Ablation 套件（§F.6 表 10 项）+ 论文写作

最高 ROI ablation 排序：
1. 删 L_HCS → 回答最大赌注（§F 选择 3）
2. γ_const = 1.0 vs γ from [C5] → 回答 "γ 这个软门控是否冗余"
3. 用可学 g 替换 delta-rule → 回答 §3.2 claim
4. 用合并平均替换 LRU → 回答 Bug 1 修正必要性
5. 单 bank vs 双通道（控制总容量）→ 回答 HC-7/§3.4 设计选择

### Stage 切换的 kill 条件

- M1 E1 IoU < 0.3：项目从 full 版本切 MVP
- 训练第 1 epoch 后 γ_sem 塌缩 ≈ 0.5 ± 0.05：EMA warmup / sigmoid 替换
- L_PRH k=64 持续不降：§3.2 claim 实证伪、回退到容量上限增加 + delta-rule 系数调整
- Slot bank top-1 占 80%+ 写入：L_CSM 没生效、加 entropy regularizer / Switch-Transformer 风格 load balance
- batch 16 上 OOM：Δ 限定 {4, 16}、接受 §H Trade-off 7 升级
- 5-loss balancing 第 8 周仍 sweep 不稳：用 GradNorm / PCGrad，否则架构整体不可控

### 显式不会做的事

- 不会基于这份架构整体训一个完整 VLA。组件没通过 E1-E4 之前不进入 E5。
- 不会在 E1 失败后强行 patch [C10]——直接 fallback 到 MVP（§B.3）。
- 不会在 BridgeV2 之外的 dataset 上跑 E1（其他 dataset 没有连续 trajectory + a*_{t+64} 对齐保证）。

### γ 必要性 ablation——M6 必跑

源文档 §F.6 列出的最高 ROI 第 2 条 ablation 在这里再强调一次：**γ_const = 1.0 vs γ from [C5] vs γ from L_HCS** 三组对比。决策规则：

- 若 γ_const = 1.0 vs γ from [C5] 长程 SR 差异 < 5% → **MVP 阶段直接简化为 γ=1，删 [C5]**（[C5] 的软门控被三层自然稀疏化覆盖、边际价值不显著）
- 若 γ_const = 1.0 vs γ from L_HCS 长程 SR 差异 < 5% → **整个 [C5][C10] 链路降级为常数门控**（这意味着架构核心赌注 §F 选择 3 失败）

这条不在 milestone 里单独列时间——M6 ablation 套件的一部分。

---

## § 11. 这份方案的隐性单点故障 + 边界

读完整套设计，必须诚实指出几处**隐性单点故障**——它们不是设计缺陷，是这条路径在 first-principles 层面就内置的脆弱处。

**[C10] HCS Jacobian 在真实 expert demo 上的信噪比是项目存亡判决。** §F 选择 3 标 [弱] 不是修辞——v1 §7 的 open question "Jacobian saliency 在真实 expert demo 信噪比够不够"未被独立验证、RUDDER 是 RL adaptation。如果 E1 IoU < 0.4，整个 §3.1 后视回应路线必须放弃，架构降级为 MVP。**这不是 fallback 路径优化、是核心 claim 的塌陷**——publishable claim 从"§3.1 + §3.2 + §3.4 + HC-3"降到"§3.2 + §3.4 + HC-3"。

**Stop-grad 矩阵的工程正确性。** 4 处 sg 任何一处漏写整个 HC-3 隔离失效。这不是设计层面的脆弱、是**实现层面**的——必须有 unit test 验证 sg 在正确位置；任何重构都要重新审查 sg 矩阵。审查里 Bug 4（γ 在中间层而非最外层 + sg）就是这种事故的具体形式。

**LRU 丢弃错误条目永久不可恢复。** delta-rule 不衰减的代价是"丢了就真丢了"。如果 [C12] CSM 评分错误（把任务相关 slot 评为低重要度），LRU 丢的就是关键事件——比合并平均的"信息模糊化"在某些任务上更糟。**何时不可接受**：event 密度高 + 长 episode（K_s = 64 频繁触发丢弃）。当前架构 episode 间清空 + 短-中 episode 规避了这条，但跨 episode 共享路径必须重新设计 LRU。

**部署时 γ 仅有代理信号。** 训练时 [C10] 把"未来需要"推入 γ，但部署时只有 self-supervised prediction error。Test 任务的 e 分布偏移会导致 γ_*_t 失准。**何时可接受**：训练数据覆盖 test 视觉/语义分布（域内 evaluation）；**何时不可接受**：跨 dataset zero-shot、对抗 distractor。这是 §H Trade-off 1 显式承认的——不是设计缺陷、是 hindsight 监督的边界条件。

**HC-5 部分回应——独立技术审查 Issue 2 把这条戳得更深**。原文档把 Jacobian + RUDDER 描述为"同组件 fork"——审查指出这条说法**仍然太轻**：J 和 RUDDER LSTM g_θ 不是并联两个独立信号源、是被同一个 z-score + sigmoid 融合**串联**喂给唯一的 ψ。三件东西（J、RUDDER、ψ）在 forward 上是 series 不是 parallel——上游信号在喂到 ψ 之前已经塌成单一 γ̂ target、ψ 是单一 student、deploy-time γ 是单一 ψ output。**这跟 Mem-0 learned classifier 单点的结构是同型的、只是中间多塞了一道蒸馏**。把 J + RUDDER 称为 "ensemble" 在结构上是错的。

**所以 HC-5 的诚实评级应该从"部分回应"再降一档**：当前架构在 HC-5 上是"延迟单点"（把分类器单点换成 saliency head 单点 + 加了一道蒸馏）——风险只是从 single-point classifier 漂移到 single-point saliency-target。**何时不可接受**：E1 失败 → 直接升级为"完全没回应 HC-5"。修复路径有两条：(a) 真正的 ensemble——训两个独立 ψ_head 分别对 J-only 和 RUDDER-only target 学习，deploy 时混合输出；(b) 把 §0 四针眼里的 HC-5 退到"未来工作"、不算在当前 claim 里。**MVP 暂时按 (b) 走**——publishable claim 不包含 HC-5 回应。

**HC-6 实时性勉强满足（§H Trade-off 8）。** 70-135 ms on H100 BF16 batch=1（纸面估算，需实测）；A100 上接近 110 ms 可能溢出；4090 48G 上 230-520 ms（2-4 Hz）—— **HC-6 在 4090 原方案下被破坏**。优化路径（ViT-B + 1-step flow + INT8 + GRU-ψ + KV cache）能压回 70-140 ms 区间，但代价是基础 BC 性能上限被 backbone 选择封顶（ViT-B vs ViT-L 在精确抓取任务上 -3-5% SR）。**这条不是隐性单点故障，是硬件锁定后的硬性约束**。

**HCS Jacobian 计算成本 + train/deploy 路径分裂（§H Trade-off 3）。** L_HCS Jacobian × 3 个 Δ + L_CSM 4 slot leave-one-out × 2 frozen forward 让单 step 计算量从 baseline 的 ~140 GFLOPs 涨到 ~1045 GFLOPs——**训练 pipeline 比纯 BC 慢 2-3 倍**。部署时这些都不跑，但 train-only / deploy 路径分裂是工程负担——需要严格 conditional forward 与 checkpoint 分组。何时可接受：≥4×H100 或 6×A800、训练预算 1 周；何时不可接受：单 GPU 或训练时间敏感场景。**fall-back 是 MVP 简化版**（disable [C10][C12][C13]，单 step 回到 ~250-400 GFLOPs）。

### 两条单点故障的优先级

E1 通过（HCS 信噪比）是数据/方法层面的天花板——一旦糊就实验闭环救不回来。stop-grad 矩阵正确性是实现层面的工程要求——可以靠 unit test 兜住但要严格执行。

**所以验证顺序是**：先做 M1 E1 sanity check（手标 100-200 BridgeV2 traj 验证 IoU @ 0.3 ≥ 0.4），通过了再投预算调整个 architecture。把这两件事弄反，会浪费大量 GPU-time 在一个 hindsight 信号信噪比不足的训练上调参——而调参永远调不到天花板之上。

---

除此之外，这份方案不解决：

- **跨 episode 知识依赖**（同一物体的物理属性、跨任务经验积累）。当前架构 episode 间清空 M_sem——是 §H Trade-off 2 显式 disclaim 的边界。
- **物体身份重排**（red ↔ blue 互换）。slot 通过 content-based key 分配，身份混淆物体可能被路由到同一 slot（hash collision）。需 SymObj 类的物体级初始化（v1 §5.2 类 D 拒绝路线，本方案未涵盖）。
- **极长 episode（>500 步）+ 高 event 密度**。K_s=64 不够 → LRU 频繁丢弃。需 K_s ≥ 128 + ensemble multiple bank。
- **反向因果场景（柔性物体接触延迟反应）**。[C13] 是兜底，MVP 默认砍。若反向因果 > 5% 帧需重新设计。
- **Distractor-heavy 视觉变化大但任务无关**。e_geo 飙高、γ_geo 错误高、grid 被无关几何写满。对策是加 distractor 增强训练数据；未对策时本架构会失败。

---

## § 12. 与现有 Memory VLA 工作的差异（一句话每条）

| 工作 | 它的核心机制 | CHIME 与之差异 |
|---|---|---|
| MemoryVLA | top-k 不可微检索 | CHIME 全可微 read，L_HCS 给旧帧梯度 |
| Mem-0 | learned subtask classifier 触发 | CHIME 用连续 prediction error + offline saliency，避免分类器单点 |
| ReMem-VLA | frozen-EMA + POP 整图重建 | CHIME 用 delta-rule（无 EMA mixing）+ MERLIN 风格 future-prediction（非整图重建）+ L_HCS 给写入端额外信号 |
| Goal2Skill | VLM verifier 慢循环 + 符号记忆 | CHIME 全在 VLA 快循环内（< 100 ms），不依赖 VLM 推理 |
| MEM (π0.6) | 视频短期 + 语言长期 | CHIME 不用语言摘要；短期=FIFO，长期=delta-rule slot |
| MemER | VLM 主动提名关键帧 | CHIME 写入端纯神经，不依赖 VLM 慢推理 |
| PSM-CWM (原型) | HMD loss (Jacobian saliency) | CHIME 把 HMD 扩展为 Jacobian + RUDDER 双 fork 信号（同组件内部 fork、§H Trade-off 6 承认仍是单点）+ 配双通道写入 + delta-rule 不可变 bank |

---

## § A. 与 v1 (CW-Mem) 的延续与差别

**延续**：不可变 bank、双层放置（结构性分离）、后视因果显著性损失。

**差别**：

- v1 写入触发器是"反事实 surprise"（模型自跑两遍对比）。**v2 改为 event-segmentation 预测误差（连续标量）+ HCS offline saliency（离线监督）双信号**——v1 的反事实 surprise 自身需训练、撞 HC-3。
- v1 的"双层放置"是 VLM 端 + action expert 端。**v2 改为 geometric 通道 + semantic 通道**——按精度分而非按位置分，理由是 §3.4 真正的分轴是精度。**这条 claim 强度 [中]**（§F 选择 2），单 bank 双 head 仍是合理 fallback。
- v1 没有显式的"working buffer / 短期 / 长期"三池区分。**v2 引入 FIFO 工作缓冲（可微）+ 不可变 slot bank（delta-rule + CSM-importance LRU）+ 几何 grid（delta-rule + timestamp eviction）**。两条容量管理机制不同：slot 按 CSM 重要度丢，grid 按 read-timestamp 丢——共用 "LRU" 名义但判据不同（命名上 grid 那条改为 timestamp eviction 以避免混淆）。
- v1 §5.1 判断四的"语义信号在子任务边界附近密集 / 感知信号在状态突变附近密集"finer-grained claim，v2 中**简化为同一个 ψ + 两条独立 projection head + 独立 EMA**——若实证不够，fall-back 是双 ψ（参数 +10M）。这是 v2 对 v1 的有意降级，因为"分布密度模式不同"的实证强度不够支持额外参数。

---

## § B. 在硬件上的可实施性（6×A800 训练 / 4090 48G 推理）

> 这一节的所有数字都是基于 FLOPs + 显存带宽的纸面推算，需要在目标硬件上实测确认。

### B.1 训练侧：6×A800-80GB 可行

A800-80GB ≈ A100-80GB BF16 算力（~280-310 TFLOPS），NVLink 带宽限制 400 GB/s。**A800 单卡 ≈ H100 单卡的 0.55-0.62 倍**；显存上限相同。6 卡聚合 1700-1860 TFLOPS（70-75% DDP 效率），约原 4×H100 计划的 0.85-0.93 倍——**基本相当，够用**。

单 step 1045 GFLOPs（L1 + L_HCS Δ ∈ {4,16} + L_PRH + L_CSM），A800 effective 100 TFLOPS/卡 → 1.4-1.8 s/step（H100 上 0.8-1.0 s）。Epoch 时间 ≈ 22 h（H100 是 20 h，6 卡多但每卡慢）。**1 周 = 7 epoch**。

**前提**：Δ=64 必须先在小 batch 上验证 OOM、gradient checkpointing 必须开、batch size 上限可能从 32 降到 24。

### B.2 推理侧：4090 48G 在原方案下大概率超时

4090 BF16 ~83-100 TFLOPS、memory bandwidth ~1 TB/s——**单步延迟约为 H100 的 3-4 倍**。原方案纸面估算 230-520 ms（2-4 Hz），**HC-6 在 4090 上被破坏**。

优化路径让 4090 单步压到 70-140 ms（7-14 Hz）：
- [C1] SigLIP-ViT-B 替代 ViT-L（-60-130 ms，代价 ViT-B vs ViT-L 在精确抓取 -3-5% SR）
- [C9] ODE 1-2 步替代 4-8 步（-70-180 ms，consistency model 蒸馏）
- 量化 [C1][C9] 到 INT8（-30-50 ms）
- [C5] ψ 简化为 1-layer GRU（-10-30 ms）
- M_sem / M_geo KV cache between steps（-5-10 ms）

**MVP 必须从 ViT-B 起步**——所有"基础 SR"对照实验跟着重新跑。

### B.3 修订后的 MVP 配置

**硬件锁定**：6×A800 训练 / 4090 48G 推理。

**默认 backbone**：[C1] = SigLIP-ViT-B + LoRA r=16；[C9] = π0 flow matching + 1-step consistency 蒸馏；推理 INT8。

**砍掉**：[C10][C12][C13]、L_HCS、L_CSM、L_GC；[C6] 单分辨率 16³；benchmark 收敛到 LIBERO-Long + CALVIN ABCD→D。

**保留**：[C1 ViT-B][C2][C3 简化][C4][C5 仅 prediction-error self-supervised, GRU 实现][C6 16³][C7 简单 timestamp LRU][C8 + KV cache][C9 1-step] + L_main + L_PRH + L_aux。

**预期**：推理 80-130 ms（7.7-12.5 Hz）✅；训练 10M frames、batch 24、6×A800、5 epoch ≈ 4-5 天。

**MVP publishable claim**："Event-segmentation prediction error 触发的双通道 delta-rule 不可变 memory + MERLIN 风格 predictive read，**推理 7-12 Hz on consumer-grade GPU**（4090 48G），在 CALVIN ABCD→D + LIBERO-Long 上 SR 比 OpenVLA + history baseline 高 X%。" —— **加入"消费级硬件可部署"作为 contribution 之一**。

---

## § C. 这个方案到底"赌"什么

不是赌"新架构一定 work"，而是赌三件事各自的概率：

1. **§3.2 真的可以靠 delta-rule + LRU 丢弃绕过**（§F 选择 1，强度 [中]）。如果错，长程任务塌陷，fallback 到类 B 的可学 g。
2. **HCS Jacobian 在真实 expert demo 上信噪比够**（§F 选择 3，强度 [弱]）。**E1 的核心赌注**。如果错，disable [C10][C12][C13]，架构降级为"prediction error self-supervised + L_PRH"的简化版，仍能打 HC-3 + §3.2 + §3.4 三条，workshop paper 级 publishable。
3. **按精度分通道（grid + slot）优于按位置分（VLM + expert）或单 bank 双 head**（§F 选择 2，强度 [中]）。如果错，合并为单 bank，损失 ~10-15% 长程 SR 但架构整体不塌。

赌注按"如果错了，会降级到什么状态"排序：**赌注 2 错 → MVP fallback 路径（§B.3）；赌注 1 错 → 整个 §3.2 claim 推翻、需要重新设计；赌注 3 错 → 单通道版本仍 publishable**。E1（M1，week 4）是触发降级路径的判决点。

---

读完整套方案的一句话总结：**CHIME 不是"用更复杂的记忆机制"，是"用更简单的代数性质（加性更新）+ 离线已知未来反推（hindsight）+ 按精度结构性分通道，把长程 VLA 问题从一个端到端 BPTT 不可达的硬约束，转换成一个可以模块化分别验证的工程问题"**。每个模块在堵某条 HC/§ 的针眼，拿掉任何一个对应的针眼就漏。
