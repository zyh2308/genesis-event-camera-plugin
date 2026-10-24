# E-POSE 服务器导入

状态：导入完成。此页由本次一次性导入流程自动刷新；不是训练或真实评测结果。

- 本地原始ZIP：`E:\EPOSE_Downloads\28106717.zip`，29,759,156,265 bytes。
- ZIP SHA256：`81fd5ef872ad5551bd170b0451eb3df3e4536a7634202b431d322e05d60e5884`。
- g1根目录：`/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1`。
- 原始压缩包：`archives/*.rar`；数据：`extracted/<子包名>/`；逐文件完整性清单与收据：`manifests/`。
- 优先导入original_data.rar；其RAR头清单列306个HDF5、21,744,552,016解压字节。其余17包补齐逐物体和多物体标注/派生数据。
- 原始外层ZIP保留在E盘；以其18个原始RAR成员上传，先验证ZIP成员CRC，再验证本地/服务器子包SHA256，原始HDF5记录每文件SHA256；其余包由官方UNRAR校验CRC且逐文件核对大小，清单明确标记校验类型。
- 实时状态：`ingestion_live_status.json`；全量完成后生成`ingestion_receipt.json`。
- 当前上传子包：`None`；已上传18/18；服务器解压阶段：`complete`，已完成18/18。
- 服务器操作日志：`/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1/logs/unrar_recovery.log`；本地日志：`E:\EPOSE_Downloads\server_transfer\supervisor.log`。

本次只传输、完整性校验、解压和生成清单，没有修改模拟器或训练配置，也没有运行真实数据评分。后续先审核HDF5/MAT、K/位姿/时间/CAD对应；数据导入完成不等于位姿模型已通过可用性验证。原始发布树保持原样，不使用GT mask/segmented events作为模型输入。

已完成文件数：2377982，解压总字节：48610498574。
