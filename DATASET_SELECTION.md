# 整图实验数据选择与准入决策

更新时间：2026-09-25。这里只记录官方来源页面可核验的信息；数据能否实际获取、发布或用于特定研究，仍以数据提供方许可和实际审批为准。

> **历史方案说明（2026-09-27）：** 本文早于当前实验执行，保留为数据选择决策记录，不代表当前项目仍在等待 Columbia 全图集或把 NIST OpenMFC 作为已完成数据。当前实际使用数据、来源记录、许可状态及公开限制以 [`data/SOURCES.csv`](data/SOURCES.csv) 和实验报告为准。NIST OpenMFC 未纳入 v1.0.0 结果。

## 结论

首选申请 **NIST OpenMFC 的 development/resource 数据**作为整图主实验数据。它的任务与本论文最接近：整图 manipulation detection/localization，部分开发资源带 references、原始高 provenance 图像、probe、操作 journal/annotation。申请账号及签署许可须由项目负责人/用户本人完成。OpenMFC 页面称项目评测已结束，因此不得假定注册后一定能拿到旧数据；若无法获批，记录往来/状态并启用备选，不使用未授权镜像。

同时继续尝试 Columbia 的未压缩彩色全图集作为小型定位桥接集。当前官方页面说明其有 183 张 authentic 和 180 张 spliced 图像，提供人工 edge masks，但此前网页指向的 Dropbox 暂时不可用。该集合以跨相机拼接为主，样本和场景覆盖较窄，且 masks 的语义是来源相机区域/边缘区；必须在协议中明确如何转换成篡改区域，不能不加说明地当作通用编辑掩码。

**不能把已经获取的 Columbia 2004 图像块集当作整图替代品。**它只用于 patch 级流程 sanity check。

## 候选数据源对照

| 候选 | 可支撑的实验 | 获取与限制 | 决策 |
|---|---|---|---|
| NIST OpenMFC development/resource | 整图篡改检测/定位；部分含参考 mask、base/probe 和 manipulation journal；可研究来源/操作变化 | 需注册并完成 data license；evaluation reference 可能封存；需确认旧资源仍可申请及公开发表条款；官方提醒不得未经书面同意公开横向比较参赛者排名 | **主申请项** |
| Columbia Uncompressed Image Splicing Detection Evaluation Dataset | 小型整图拼接检测；官方提供 edgemask，可研究拼接边缘/来源区域 | 183 authentic、180 spliced；下载表单/Dropbox 曾用但当前链接受阻；小而旧、内容和操纵类型单一；mask 解释需谨慎 | **并行尝试的桥接集** |
| Columbia Image Splicing Detection Evaluation Dataset (2004 image-block) | 128×128 图像块二分类流程检查 | 1,845 blocks；research use only、禁止再分发；缺少原始整图/场景 ID | **仅 sanity check，不做核心证据** |
| GenImage | 真实照片与多生成器全合成图检测、generator holdout | 不解决局部编辑掩码/整图局部定位；按官方版本和来源条款核查 | **用于独立的全合成检测子任务** |
| SciSp-H / SciSp-C | 科学图像拼接痕迹/特定场景探索 | Zenodo 官方记录说明 SciSp-C 因未取得各出版方图像版权而仅提供详细标识符；不是通用自然场景基准 | **可选专项外部验证，不作主集** |

官方入口：

- NIST：[OpenMFC](https://mfc.nist.gov/)；[NIST OpenMFC 项目页](https://www.nist.gov/itl/iad/mltg/open-media-forensics-challenge)
- Columbia 全图集：[官方数据集页](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)；[官方获取说明](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html)
- Columbia 图像块集：[官方说明和禁止再分发条款](https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm)
- GenImage：[官方仓库](https://github.com/GenImage-Dataset/GenImage)
- SciSp：[Zenodo 官方记录](https://zenodo.org/records/10989921/latest)

## 数据准入门槛

每个数据源在训练/评测前必须补齐 `data/SOURCES.csv` 条目：官方版本和发布日期、下载/申请路径、提供方许可证/协议原文、允许训练和论文发表与否、是否允许上传样本索引/哈希/预测结果、原始标签解释、图像/掩码对齐验证、source/base/donor/probe 字段、数据划分字段、引用/致谢要求、存放位置及审批状态。

拿到 NIST 数据后，先仅下载/检查提供的 index、README、reference 说明和许可；确认 development/reference 子集和公开条款后再获取完整图像。建立映射时保留官方原始 ID 在本地受控清单，公开稿只使用许可允许的匿名标识。按 manipulation journal/base/source family 分组切分；同一 base、donor、journal 及其全部派生图必须在一个 split。NIST sequestered evaluation 数据只能按其规则使用，不能自行猜测标签。

对 Columbia 全图集，解析文件命名的相机/场景关系，按 source scene/camera pair 建组；核对原图、篡改图、edge mask 及官方提供的 `_edgemask_3` 的像素含义，先由两人/规则化脚本复核若干样本再冻结二值 mask 映射。由于 authentic 样本也有“可疑边缘”标记，它们不能误作正类篡改区域。

## 执行顺序

1. 由项目负责人申请 NIST OpenMFC development/resource 数据访问，并留存许可/账号获批状态；此项需要真实机构/个人信息，代理不能代填。
2. 同时向 Columbia 页面提交全图集申请或联系联系人，等待 Dropbox 修复/可用下载。
3. 哪个数据源先获得合规访问，就先构建源图级 manifest、mask 校验和小规模固定 baseline；不得因先拿到数据而改变主要研究问题。
4. 至少形成两个互补测试轨道：整图局部操纵检测/定位（NIST/Columbia）与全合成检测（GenImage）。分开报结果，不混成一个总准确率。
5. 有许可且数据结构核验完成后，注册固定拆分和实验分析，再做模型比较、shift/stress、evidence replay 和失败审计。

如果 NIST 和 Columbia 全图访问均失败，暂停“真实整图定位”主张，先把论文收窄为方法/协议与 patch 级 pipeline demonstration；不从未授权的第三方镜像拼凑主结果。
