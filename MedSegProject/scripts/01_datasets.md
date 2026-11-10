# 数据集：下载地址、目录规范、注意事项

所有数据放在 `$WORK/data/`，每个数据集一个目录，里面统一成 `images/` 和 `masks/` 两个子目录，**图像与掩码同名（扩展名可以不同）**。`02_make_manifests.py` 按这个规范扫描。

## 息肉（结肠镜）

| 数据集 | 张数 | 来源 | 备注 |
|---|---|---|---|
| Kvasir-SEG | 1000 | https://datasets.simula.no/kvasir-seg/ | 原始包已是 images/ masks/ |
| CVC-ClinicDB | 612 | https://polyp.grand-challenge.org/CVCClinicDB/ | 原始包是 Original/ 与 Ground Truth/（tif），脚本能识别 |
| CVC-ColonDB | 380 | PraNet 测试包（见下） | |
| ETIS-LaribPolypDB | 196 | PraNet 测试包 | |
| CVC-300（EndoScene 测试子集） | 60 | PraNet 测试包 | |
| **PraNet 划分包** | 训练 1450 / 测试 798 | https://github.com/DengPingFan/PraNet（README 里的 TrainDataset / TestDataset 云盘链接） | **强烈建议用这个包**，训练集 = Kvasir 900 + ClinicDB 550，测试集五个，与几十篇文献完全同一划分 |
| **PolypGen** | 1537 单帧 + 序列帧，6 个中心 | https://www.synapse.org/#!Synapse:syn26376615（需注册并同意协议） | 目录 `data_C1…data_C6/images_Cx, masks_Cx`，掩码文件名带 `_mask`。中心编号就是域标签 |

PraNet 包解压后放成：
```
data/PraNet/TrainDataset/{images,masks}
data/PraNet/TestDataset/{Kvasir,CVC-ClinicDB,CVC-ColonDB,ETIS-LaribPolypDB,CVC-300}/{images,masks}
```
脚本会自动把 TestDataset 下的五个目录当成五个数据集。

## 皮肤（皮肤镜）

| 数据集 | 张数 | 来源 | 备注 |
|---|---|---|---|
| ISIC2018 Task 1 训练集 | 2594 | https://challenge.isic-archive.com/data/#2018 | 图像在 `ISIC2018_Task1-2_Training_Input`，掩码在 `ISIC2018_Task1_Training_GroundTruth`（`_segmentation.png`） |
| ISIC2018 Task 1 验证/测试集 | 100 / 1000 | 同上 | 测试集真值 2019 年后已公开，可作为同分布测试 |
| PH2 | 200 | https://www.fc.up.pt/addi/ph2%20database.html | 目录 `IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp` 与 `IMDxxx_lesion/IMDxxx_lesion.bmp`，需先整理成 images/masks（手册里有一行命令） |
| ISIC2017 测试集 | 600 | https://challenge.isic-archive.com/data/#2017 | 可作第二个皮肤目标域，**但与 ISIC2018 训练集有重叠，必须先跑 06 去重再用** |

注意：Derm7pt 没有分割掩码，不能当分割目标域（评审报告里提到它是我的疏漏，这里更正）。

## 目标域泄漏核查必须覆盖的"训练池"

大论文写第二阶段用了 "SAM-Med 数据集的 0.1% 子集"微调。SA-Med2D-20M 汇编了 ISIC、Kvasir、CVC 等公开数据。向项目组要到那个子集的**文件清单或原图**后，与 ColonDB / ETIS / CVC-300 / PH2 / PolypGen 各中心跑 `06_dedup_check.py`，结果写进论文的 Data 一节。

## SAM 系权重

| 模型 | 权重 |
|---|---|
| SAM ViT-B | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |
| SAM ViT-H（教师） | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| MobileSAM | MobileSAM 仓库 `weights/mobile_sam.pt` |
| EdgeSAM | EdgeSAM 仓库 README（edge_sam.pth / edge_sam_3x.pth） |
| RepViT-SAM | RepViT 仓库 sam 子目录 README |
| EfficientSAM-Ti / S | EfficientSAM 仓库 `weights/` |
| MedSAM | MedSAM 仓库 README（medsam_vit_b.pth） |
| SAM-Med2D | SAM-Med2D 仓库 README |

权重统一放 `$WORK/ckpt/`。
