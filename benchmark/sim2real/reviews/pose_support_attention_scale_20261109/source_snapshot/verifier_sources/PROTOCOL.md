# 固定 k=12 注意力图独立复核

这是原完整诊断之外的独立零更新 verifier，不改已经冻结的 V1 或设备修复 V2，不重训、不评分真实数据。样本在读取本轮新结果之前由根会话固定：每条轨迹只取窗口 k=12，覆盖四条轨迹、三光照、两生成器、三 seed、C0/C1、独立 center/depth 分支及 RAW/BAF 两输入，共576张注意力图。

复用已锁 BAF helper `D.prepare` 得到原 encoder416→原 AREA128 输入；不复制 encoder。导入该 CPU helper 前后须 CUDA 未初始化，保护并恢复原队列分配的 GPU4 UUID，随后仍由原 `E.check_gpu` 检查 UUID、可见卡数与排他 PID。若 helper 初始化 CUDA，则直接拒绝，不能修改已初始化上下文。CPU 模式必须显式隐藏 CUDA。helper 自带的有序 identity 检查保证各光照窗口 ID、K/T/keys/UV 对应。

原模型只注册 feature/center-logit hooks，原 forward 完整运行。独立计算 `softmax(logits) @ 原grid`，与同一次原 forward 的 UV 精确相等；模型及 buffer、源模型文件、输入和依赖文件哈希前后保持不变。按根会话在执行前的要求，保留 producer 每组64窗的原顺序与batch32形状，实际前向9216个输入、288个batch，再仅抽取预定k=12的576图核验/保存，不以batch4引入无必要的数值因素。其他窗口虽经过原前向，不宣称已独立审核它们的完整注意力图。

从 H5 直接寻找21帧 target world pose 对应的唯一 object slot，再取得真实 segmentation ID；独立核21帧可见像素计数与目标区域 depth，并读取 k=12/13 端点区域及其并集。端点区域只在该帧精确；并集仅为近似，不能代表整10ms事件时刻的可见性。所有 GT 都只用于 forward 之后的区域统计、身份/时间核验，不作为模型输入、裁剪或选点。

数值 oracle 不调用 `S.summarize_attention` 或其他 support 数值 helper。独立执行 native mask→OpenCV LINEAR416→Torch CPU AREA128→AREA16，占比作为空间代理，并计算注意力质量、面积富集比、峰值与 UV 的 native floor-pixel 命中。空区域保留 null，不删样本。

预先固定比较规则：原保存 UV 绝对误差≤2e-6；区域注意力 mass 绝对误差≤2e-6；富集比误差界为2e-6/区域面积占比；区域面积占比误差≤1e-12；峰坐标与离散命中严格一致。若恰在数值并列或像素边界发生差异，保留失败记录和新原始 map，不扩大容差。对比只证明预定576图的复核，不能外推为全部9216图已独立重算。

输出完整576张FP32 map NPZ、索引与去重后的 native bool mask NPZ，以及逐例差异和小 completion 收据。完整 map 文件保留服务器，本地可只归档收据和固定少量数组。2×2 hybrid 全6144行的数学/宏汇总另由已保存 full 结果独立审计，不纳入该 verifier 的模型前向范围。

安全上限1200秒；CPU/GPU4及队列由根会话决定。此处只准备源码，不连接服务器、不执行 verifier 或 fixture、不启动任何训练。七项 CPU fixture 已提供，包含实际64输入分两组32的 toy-hook 前向，待根会话实际执行后方可声明通过。

```text
python3 -B -m unittest -v test_attention_sentinel_verifier
python3 -B verify_attention_sentinels.py --source-manifest-sha256 <verifier_source_SHA> --full-completion <full/completion.json> --full-completion-sha256 <SHA> --attention-sources <V2_sources> --diagnostic-sources <frozen_E_sources> --baf-sources <frozen_BAF2_sources> --support-sources <frozen_support_sources> --contract <original_contract.json> --device cuda --out <fresh_output>
```
