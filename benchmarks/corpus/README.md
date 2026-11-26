# L20 字幕样片

`corpus.json` 固定了 15 个独立原片的起止帧、输入摘要、字幕选区和复核帧。主集共 1,794 帧、约 60.475 秒。每个原片只使用一个片段，时长为 3.47 至 5 秒。

| 样式 | 调参 | 留出验收 |
| --- | --- | --- |
| 普通字幕 | c01、c02、c03 | c04、c05 |
| 粗体或粗描边 | c06、c07、c08 | c09、c10 |
| 粗体或明显粗暗边，带底板 | c11、c12、c16 | c13、c14 |

c01 覆盖英文短剧中原先漏掉的单词、逗号与双行词尾。c06、c07 取自两条历史粗字残影原片。c08 保留原片第 215 至 334 帧，其中样片第 90 至 119 帧对应原片最后一秒。

c11 是中英文两行各自带黑底的讲解，c12 是叠在黄色笔刷底上的粗描边文字。c16 直接取自 Captions 已公开发布的 Bulletin 样式成片，画面中的粗黑字分行落在不透明白色字卡上。[Captions](https://captions.ai/)

c13 和 c14 分别取自两部 NASA 戈达德作品。c13 的白色粗体叠在半透明紫色底板上，c14 有突出的黑色偏移描边和紫色圆角底板。两者共享制作机构和出镜人，均归入留出验收。NASA 同时发布了逐帧对应的无字幕版本，清单内的 `reference` 指向同一时间段的无损截取。原始发布版本各自经过视频压缩，字幕区外的画面仍有小幅编码误差。[光谱学短片](https://svs.gsfc.nasa.gov/14801/) · [黑洞结构短片](https://svs.gsfc.nasa.gov/14905/)

## 补充样片

`supplemental.json` 记录两个独立的补充案例。c15 是 Natural Resources Wales 发布的半透明黑底、中等字重字幕。p01 是 MFVSR 在真实高画质素材上制作并发布的配对基准，附有官方渲染前原片。其输入与参考保留了发布方的视频文件字节。两例均单独统计。[Natural Resources Wales 原片](https://commons.wikimedia.org/wiki/File:Nature_and_Us_-_Iolo_Williams_promo_video_with_subtitles.webm) · [MFVSR 数据来源与授权](https://huggingface.co/datasets/siiian/MFVSR/blob/281a094dbe81464730aaaf2c3063e0357bfd10bc/DATA_SOURCES.md)

`extras.json` 记录两个调度输入。x01 长 4 秒，同时带有中下方字幕和右下角水印，`regions` 分别提供 `subtitle` 与 `watermark` 选区。x02 长 20 秒，可与 x01 混合提交。两者复用 c03 的调参原片。

## 本地校验

媒体保存在本地样片目录。与每次运行绑定的元数据包括原片和片段的 SHA-256、原片时间范围、编码画面像素坐标与零起算的显示帧号。原片帧区间以及选区右边和下边均为开区间。运行以下命令可核对样片摘要、真实时长、解码帧数、选区及配对片时间。命令需要 Python 3 和 FFprobe。

```sh
python3 benchmarks/corpus/verify.py --media-root "/path/to/local-bundle" \
  benchmarks/corpus/corpus.json benchmarks/corpus/extras.json benchmarks/corpus/supplemental.json
```
