# Census结构鲁棒性配对消融：归档与当前门禁状态

本目录 sources/ 原字节归档了最终13个成员及原 manifest；清单SHA为
`45343592cc7eff1c86f64bafe0d91362f96371f84071daf19ae32cff9060ff04`。
冻结源码及其中的构建时说明保持原样；本页补充随后实际执行状态，不回写源码或旧结果。

## 当前已确认和未确认的内容

研究主会话已在g1实际完成科学代码CPU8/8测试和mode cpu集成核验：
冻结源码、82项参数清单、64份原始数据灰度域、两份last8910检查点均通过。
这与 sources/cpu_preflight.json 中较早的本地CPU夹具记录是两份不同证据；
后者未被替换成服务器结果。本次归档没有重新SSH执行核验。

主会话已在g1 GPU3实际确认Census phase0任务153退出码0：
两种生成器各8个固定TRAIN批次，共16个批次，optimizer.step次数为0；
82项完整参数签名与70项dt1连接梯度检查均通过。
统一冻结的Census系数为lambda = 0.00018879906228896475，
由预登记的16个参数梯度范数比例的中位数倒数计算。

上述数值来自主会话对服务器的实际核验反馈。本页尚未收到完整phase0
机器证据导出，因此该lambda尚未由本子会话独立重算16项比例。
四臂GPU一步smoke任务154已启动，完整四更新/恢复收据尚未取得；
不能称smoke或200步study已通过。真实评分仍关闭，等待后续GPU门禁。
未来权威产物位于独立实验根
/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-census-paired-short-v1/：
cpu/completion.json、phase0/completion.json与calibration_lock.json、
smoke/completion.json、study/completion.json。目录/文件名本身不等于已经完成。

## 这项实验回答什么

全模型参数空间下，原光度项与GT误差梯度的方向总体一致：
EventGenesis的NN/strict余弦约0.8886/0.8881，
V2E约0.7586/0.7589。局部活跃区域仍有4/32与3/32负向样例，
gain/offset负向控制约-0.00367，幅度很浅。
因此现有证据不支持“原光度项整体把模型训练错方向”的总根因判断。

新Census仅是一个预先声明的结构/光度鲁棒性消融，用相同模型、
原control采样与增强、完整四尺度像素SUM、原smoothness10做配对对照。
Census与预测patch support作为整体objective变化，不能单独归因某个子机制。
mask-only改变原SUM损失常数而不改变梯度，已作为负对照排除，
不应再为它启动补训。

## 零更新阶段与显存

phase0并不是“没有创建optimizer”。复用的B.restore会构造Adam与scheduler，
恢复原模型、Adam、scheduler和RNG；phase0计算autograd.grad用于校准，
但不调用optimizer.step，Adam计数应保持8910，模型参数与旧检查点不变。
完整参数集合是82项，dt1实际连接并产生有限梯度的是70项；
未连接项按零计入完整向量范数，不裁剪参数集合。

本地CPU仅对独立Census分支做saved-tensor审计，batch1四尺度
256/128/64/32、patch7的唯一保留storage为84.668MiB；
线性推算batch8约677.344MiB，单个全尺度邻域feature bundle约130.156MiB。
加上原模型保留图、两组参数梯度和临时量，结合此前报告的原训练峰值
25,566MiB，phase0合理估计约27–30GiB，32GiB卡大概率可以容纳。

上述27–30GiB仍是历史CPU推算，不是GPU峰值测量。现在phase0已在GPU3
实际完成，证明该零更新阶段可以容纳；它不能替代四臂smoke的真实
optimizer更新及保存/恢复内存门禁。尚无实际峰值数值可用于论文效率图。
retain_graph、CUDA分配器和cuDNN工作区仍可能改变后续峰值。
若后续OOM，应保留失败证据并交由研究主会话决定下一版，
不自动改batch、精度、采样、loss或公平性条件。

本次只新增归档、说明和CPU内存估计；没有修改根报告/状态，
没有Git add/commit/push，也没有启动、恢复或重跑服务器任务。
