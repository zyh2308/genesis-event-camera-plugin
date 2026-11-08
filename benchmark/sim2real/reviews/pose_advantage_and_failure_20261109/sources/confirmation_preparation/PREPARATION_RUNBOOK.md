# 32条独立共享轨迹：CPU物理生成与GPU事件重放

这是优势确认的准备阶段，不训练、不评分真实数据、不决定模型采用。科学preregistration沿用根会话已登记的`advantage_registration.json`（SHA cdb5a83e5c80798a700f5c001b93d7994ab2c1b272212e6437f80342f07c3c52）。操作合同另写，不能覆盖科学登记。

1. 将本锁包放在N下独立`advantage_preparation_sources_v1/`，N=`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261109-pose-advantage-and-failure-v1`。根会话核source_manifest每个真实字节，执行4项CPU测试。
2. 在服务器CPU运行`build_preparation_contract.py --relight-source <本锁包/prepare_pose_radiance_pairs.py> --out N/PREPARATION_CONTRACT.json`。它读取实际N的3份JSON，绑定实际source/旧C1/基线分数和17文件core冻结收据，写独立操作合同；没有GPU调用或模型前向。根会话复审合同并固定其SHA。
3. CPU物理队列以`CUDA_VISIBLE_DEVICES='' python -B -u prepare_confirmation.py physical --registration N/PREPARATION_CONTRACT.json --registration-sha256 <实际SHA>`执行。根parent提供有界timeout/只清理自身进程组。CPU阶段不申请GPU、不导入GPU guard。AST逐句复用原generate前半段；只是把最后原事件重放loop延后。输出固定N/advantage_confirmation/normal/pose_bank；完成收据为`physical_completion.json`。绝不覆盖旧数据；失败保留独立failure，不自动重跑或换seed。
4. CPU完成后，GPU4现有资源父队列准入且绑定实际physical收据SHA，运行`prepare_confirmation.py events --registration <合同> --registration-sha256 <SHA> --physical-receipt-sha256 <实际SHA>`。原E.check_gpu/UUID/PID guard照旧；先normal，再half、eighth，每条件按EventGenesis、V2E原argv重放。光照只复用原relight函数，不改变物理标签/噪声/事件core。完成收据`event_completion.json`。
5. evaluate将作为后继独立锁包，不改本包源字节，不因预备阶段成功就宣称优势确认通过。

物理32条、新base202611090137，模型固定RAW C1，三seed和三光照全保留。statistics bootstrap seed沿**实际科学登记20261109137**，与物理base是两个不同整数，不能混用。原seed库存实际为3个注册协议、226 H5、226个scene seed，已经无新32重叠；脚本逐项复核该明确范围，**不声称全盘穷尽**。

source_records覆盖冻结R/sources的生成worker、两replay、profiles/pins/pose配置/recipe及benchmark全部Python文件。核心按已有17 canonical-LF文件冻结收据及实际原始SHA前后复核，快照没有.git不会用HEAD误判。生成源link只连冻结R/sources，唯一新pose配置只替换count/seed/worker这三项。

本地WSL CPU4/4实际通过；尚未运行实际physical/events。首个Windows测试曾因fixture使用Linux绝对路径而失败，已仅修测试路径，Linux实际合同语义未放宽。AST测试有随包原fresh source fixture，服务器不依赖本地兄弟目录。事件阶段仍须根会话进行实际设备/资源准入。
