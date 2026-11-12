# 视频结果

| 文件 | 说明 |
| --- | --- |
| [12 秒竖版](final/portrait-12s-model.mp4) | 从原片整帧推理的输出 |
| [横版优化结果](final/landscape-optimized.mp4) | 收紧选框后的输出，含第 10 秒附近单字的重新推理结果 |
| [15 秒竖版局部重跑](final/portrait-15s-retuned.mp4) | 3.30–4.23 秒按镜头从原片重跑，其余时段沿用当时的输出 |
| [15 秒竖版按镜头](final/portrait-15s-shot-aware.mp4) | 现有按镜头流程从原片整段一次导出的验证结果 |
| [12 秒竖版左右对照](comparisons/portrait-12s-side-by-side.mp4) | 左为原片，右为整帧模型输出 |
| [横版左右对照](comparisons/landscape-side-by-side.mp4) | 左为原片，右为优化后的横版 |
| [参考视频对照](comparisons/reference-gpu-samples.mp4) | 左为参考片段，中为当时的版本，右为按行合成实验 |
| [横版按镜头实验](experiments/landscape-shot-aware.mp4) | 整段单次推理结果，第 10 秒附近仍有短字幕残留 |
| [12 秒竖版早期补修](experiments/portrait-12s-early-refinement.mp4) | 早期针对片尾字幕的局部实验 |
| [15 秒竖版按行合成](experiments/portrait-15s-line-blend-baseline.mp4) | 启用按镜头分段前的输出，3–4 秒仍有拖影 |
| [横版宽选框](experiments/landscape-wide-roi-baseline.mp4) | 使用宽字幕框的早期输出 |
| [横版单字修正前](experiments/landscape-before-short-caption-retune.mp4) | 收紧选框后的第一次整段输出 |
| [竖版网页任务](experiments/portrait-shot-aware-web-job.mp4) | 网页按镜头分段实际任务的输出 |
