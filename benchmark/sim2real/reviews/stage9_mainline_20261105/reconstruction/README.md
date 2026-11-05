# Stage9 HQF五方法三seed评价桥接

这是一份独立新增适配器，不修改Stage5协议、旧scorer/infer、模型或训练。当前仅完成代码与CPU合同检查，**没有真实推理、LPIPS评分或GPU执行**。

## 交付文件

- `stage9_hqf_bridge.py`：prepare、verify和受门禁保护的run。
- `HQF_STAGE9_PROTOCOL_TEMPLATE.json`：人可读模板，不可直接执行或当评分许可。
- `test_stage9_hqf_contract.py`：只用假收据、假checkpoint字节和假数据描述的拒绝检查。
- `contract_selftest_v3/final_release/contract_test_results.json`：26/26通过。它不是实际模型、真实HQF或指标正确性测量。

旧HQF模型结构与Stage9相同，`synthetic_best.pt`包含`model`键，原推理器支持从recipe提供model_args。此桥接直接调用旧推理器；评分阶段在私有import命名空间中复用**未改动的score_pair/指标函数**，仅把循环身份列表设为15个真实`method_seed`并提供新协议路径。没有把FILM等伪装为V2E，也不改动原模块源文件或其他进程。

## 准备新协议

由主会话将三个运行脚本/模板存入独立目录。使用已登记Python和当前只读仓库；以下是服务器路径示例，其中REPO必须替换为主会话实际部署且已核对的完整benchmark checkout，不可指向缺少依赖的裸脚本目录。

```bash
P=/home/zhaoyuhan/Eventbased_WAM
R=$P/runs/sim2real/20261103-stage9-direct-v1
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
REPO=<已核对的完整benchmark仓库>
BRIDGE=<本次独立部署目录>/stage9_hqf_bridge.py
OUT=<全新协议目录>

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  "$PY" "$BRIDGE" prepare \
  --repo "$REPO" --campaign "$R" \
  --hqf-root "$P/data/sim2real/stage5/20261021-hqf14-memmap-v3" \
  --model-root "$P/external/sources/downstream/event_cnn_minimal" \
  --evreal-root "$P/external/sources/downstream/EVREAL" --out "$OUT"
```

prepare首先要求**全部15个正式训练**的completion存在、12000更新、身份/recipe/paired初始化一致、synthetic_best实际SHA匹配。缺任何一组立即失败，且不会创建协议输出目录，不接受last/preflight/短训替代。然后核对HQF14全部登记文件的大小和SHA、继承协议、模型架构、已锁源码及脚本指纹。

源目录支持两种合法形式：干净锁定git checkout，或downstream source bundle逐文件/整树哈希一致的归档副本。历史HQF专用稀疏checkout亦可在现有验证器认定干净时使用。

成功后只产生`hqf15_protocol.json`、SHA sidecar和`authorization_template_DO_NOT_RUN.json`。评分仍关闭。验证命令：

```bash
CUDA_VISIBLE_DEVICES='' "$PY" "$BRIDGE" verify --protocol "$OUT/hqf15_protocol.json"
```

verify和真实run的启动边界都会重新调用锁定源码树验证器，对EVREAL及event_cnn_minimal做全量复核；prepare通过之后再修改下游源代码也会在创建真实输出目录前被拒绝。

prepare之前不可用真实数据/分数挑选模型；全部权重按各自合成验证最低LPIPS-VGG选取。继承14序列时间完整性、事件窗、pad/crop、循环状态和输出clip合同，目标HQF属于**此前已见benchmark复测**。HS-ERGB保持extension，不能用此桥接冒充其结果。

## CPU合同检查

```bash
CUDA_VISIBLE_DEVICES='' python3 test_stage9_hqf_contract.py --out <全新假数据检查目录>
```

检查包括完整15矩阵、短预算、错recipe、真实训练flag、错方法身份、paired初始化不一致、checkpoint字节被改、用last替代、缺1组、协议绑定及sidecar篡改、两份下游源代码篡改、启动时重查源码，以及缺评分授权时输出目录创建前拒绝。另有完整推理metadata fixture及错权重、错方法、GT输入、公共预训练权重、错循环重置、缺时间端点、错源码、错数据hash等拒绝检查。没有打开实际HQF文件或Torch模型，不需要scikit-image/GPU。

## 正式评分入口和边界

主会话完成协议、权重及既有用户授权范围核验后，另存一个不可变的授权记录，内容必须绑定新协议SHA：

```json
{
  "schema": "stage9_final_real_evaluation_authorization_v1",
  "authorized": true,
  "task": "reconstruction",
  "protocol_sha256": "实际新协议SHA256",
  "source_user_instruction": "已有用户授权的准确来源或原文",
  "research_verification_note": "新15权重、HQF14暴露角色与评价协议已核对",
  "no_feedback_to_training_or_selection": true
}
```

这是防止脚本误启动的机器记录，不要求用户重复批准已经授权的工作；不能填写虚构的用户授权。模板默认false不能运行。真正启动还要求显式`--enable-real-scoring`：

```bash
# 仅在主会话取得明确GPU资源后，沿用其真实CUDA_VISIBLE_DEVICES租约
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  "$PY" "$BRIDGE" run --protocol "$OUT/hqf15_protocol.json" \
  --authorization <研究侧实际授权记录.json> --out <全新真实输出目录> \
  --device cuda:0 --enable-real-scoring
```

本子任务未运行这条命令。正式入口需要现有Torch、EVREAL、E2VID、scikit-image与登记的LPIPS模型/权重；不安装环境、不自动下载替代权重。run在推理前先确认scikit-image及本地LPIPS/backbone文件，并在CPU初始化真实metric（不评分任何图像），缺件提前拒绝。parent应先用既有合成推理/度量fixture核查该运行环境。旧评分器自己的两arm自检包含硬编码56行，不能简单改其全局身份后把该旧自检当15arm通过。

运行目录不可复用或覆盖。任何单元失败保留日志并停止；不自动跳过方法、seed、序列或坏指标。它按15run×14序列完整推理，再复用原指标循环。输出包含每帧score JSON和`per_sequence_scores.csv`，后者可送入现有`aggregate_paired_real_scores.py`。若PSNR出现数学上的+inf，CSV保留inf，旧聚合器会因有限值要求拒绝；须明确做inf兼容汇总，不能删掉这些帧或改值。

每次推理完成及评分前都核对完整event-only/模型身份/源码/循环状态/输出数量合同。原infer的原始provenance包含历史写死的e79 core，桥接保留原件不改写；新run manifest逐项记录`effective_training_core=62da…`并解释该旧字段，正式权重身份由Stage9 recipe、completion及checkpoint SHA决定。评分绑定不可变的`inference_manifest_frozen.json`，后续live run状态变化不会使其哈希失效。

目前run采用完整单次执行，不提供部分真实结果的自动续跑。失败后不得覆盖既有输出；后续若需要接续，应先校验每个已完成inference manifest/权重/输出SHA再新增独立resume适配。
