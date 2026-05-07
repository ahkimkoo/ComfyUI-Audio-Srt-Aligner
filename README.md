# ComfyUI-Audio-Srt-Aligner

基于 faster-whisper 的词级时间戳，将音频与参考文本对齐，自动生成精确时间戳的 SRT 字幕，并可将字幕渲染合成到视频画面上。

## 功能特性

- **AudioSrtAligner (文稿校对字幕)** — 将音频转换为 SRT 字幕。**参考文本为可选项**：
  - **填写参考文本**：将参考文本/台词文稿与音频对齐，生成带精确时间戳的校对后字幕
  - **留空参考文本**：直接使用 Whisper 语音识别原始结果，无需文稿校对
  - 支持中英文等多种语言，自动检测语言
  - 支持字幕最大字数限制与智能分词切分
  - 支持按标点符号切分，句中标点替换为空格，句末标点移除
  
- **VideoSrtOverlay (字幕合成)** — 将 SRT 字幕渲染合成到视频画面上。支持中文字体下拉选择、自动折行、描边阴影、淡入淡出等效果。

## 安装

1. 将本插件克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:ahkimkoo/ComfyUI-Audio-Srt-Aligner.git
```

2. 安装依赖：

```bash
cd ComfyUI/custom_nodes/ComfyUI-Audio-Srt-Aligner
pip install -r requirements.txt
```

3. 重启 ComfyUI，节点将自动注册。

## 典型工作流

![工作流截图](example/snapshot.png)

📥 [下载工作流文件 (Audio-srt-aligner--srt-overlay.json)](example/Audio-srt-aligner--srt-overlay.json)

```
LoadAudio → AudioSrtAligner → VideoSrtOverlay → Preview/Save
              ↑                    ↑
         reference_text       images (视频帧)
```

---

## 节点 1：AudioSrtAligner (文稿校对字幕)

将音频转换为带时间戳的 SRT 字幕。支持两种工作模式：**有参考文本的校对模式**和**无参考文本的直接识别模式**。

### 输入参数

| 参数 | 类型 | 必填 | 默认值 | 格式/范围 | 作用说明 |
|------|------|------|--------|-----------|----------|
| `audio` | AUDIO | **是** | — | ComfyUI AUDIO 类型 | 音频输入，来自 LoadAudio 或其他音频输出节点。格式：`{"waveform": tensor, "sample_rate": int}` |
| `reference_text` | STRING | **否** | — | 任意文本 | **可选项**。参考文本/台词文稿。填写后进行文稿校对对齐；留空则直接使用 Whisper 原始识别结果 |
| `model_size` | COMBO | 是 | `small` | `tiny`/`base`/`small`/`medium`/`large-v3` | Whisper 模型大小。模型越大识别越准但速度越慢 |
| `language` | STRING | 是 | `zh` | 语言代码如 `zh`、`en`、`ja` 等 | 音频语言。留空时自动检测，但建议明确指定以提高准确性 |
| `beam_size` | INT | 否 | `5` | 1-10 | Beam search 大小。值越大识别结果越稳定但速度越慢 |
| `max_chars` | INT | 否 | `12` | 1-100 | 每行字幕最大字数限制。超出时按中文分词边界自动切分，句中标点替换为空格 |
| `compute_type` | COMBO | 否 | `int8` | `int8`/`int8_float16`/`float16`/`float32` | 计算精度。`int8` 速度最快；Apple Silicon 建议用 `int8` 而非 `float16` |

### 输出参数

| 输出 | 类型 | 格式 | 作用说明 |
|------|------|------|----------|
| `srt_string` | STRING | 标准 SRT 格式文本 | 生成的 SRT 字幕内容，可直接保存为 `.srt` 文件或传入 VideoSrtOverlay |
| `detected_language` | STRING | 语言代码如 `zh`、`en` | Whisper 检测到的语言代码 |
| `srt_entries` | INT | 正整数 | SRT 字幕条目数量（行数） |
| `coverage` | FLOAT | 0.0-1.0 | **仅校对模式有效**。对齐覆盖率，表示参考文本与音频的匹配程度 |

### 工作模式说明

#### 模式一：有参考文本（校对模式）

当填写 `reference_text` 时，节点会：
1. 用 Whisper 识别音频内容
2. 将识别结果与参考文本对齐
3. 生成带精确时间戳的校对后字幕
4. 按 `max_chars` 限制合并/切分字幕

**适用场景**：已有准确文稿，需要给文稿加上时间戳。

#### 模式二：无参考文本（直接识别模式）

当 `reference_text` 留空时，节点会：
1. 直接用 Whisper 识别音频
2. 按标点符号切分识别结果
3. 移除标点（句中标点→空格，句末标点→移除）
4. 按 `max_chars` 限制切分字幕

**适用场景**：没有现成文稿，需要快速生成字幕。

### max_chars 分句处理规则

无论哪种模式，都会按以下规则处理字幕：

1. **标点切分**：按中文/英文标点符号切分文本为短句
2. **标点清理**：
   - 句中标点（逗号、顿号等）→ 替换为单个空格
   - 句末标点（句号、问号等）→ 直接移除
3. **字数限制**：合并短句直到接近 `max_chars`，使用 jieba 分词确保不切分词语
4. **时间插值**：切分后的字幕按字数比例重新分配时间戳

**示例**（`max_chars=12`）：
- 原文：`今天天气真好，阳光明媚，适合去公园散步。`
- 处理后：`今天天气真好 阳光明媚` | `适合去公园散步`

---

## 节点 2：VideoSrtOverlay (字幕合成)

将 SRT 字幕渲染合成到视频画面上。严格按 SRT 原文逐条渲染，不合并不拆分。

### 输入参数

| 参数 | 类型 | 必填 | 默认值 | 格式/范围 | 作用说明 |
|------|------|------|--------|-----------|----------|
| `images` | IMAGE | **是** | — | `(B, H, W, C)` 张量 | 视频帧批次输入。B=批次大小，H=高度，W=宽度，C=通道(通常为3或4) |
| `srt_string` | STRING | **是** | — | 标准 SRT 格式 | SRT 字幕内容。来自 AudioSrtAligner 或其他 SRT 输出 |
| `font_family` | COMBO | **是** | — | 字体文件名 | 字体下拉选择。自动聚合 `fonts/` 和 `ComfyUI/models/fonts/` 目录下的 `.ttf/.otf/.ttc` 字体文件 |
| `font_size` | INT | 否 | `80` | 12-200 | 字幕字体大小（像素）。数值越大字体越大 |
| `font_color` | STRING | 否 | `#FFFFFF` | 十六进制颜色码 | 字体颜色。如 `#FFFFFF`（白色）、`#FF0000`（红色）、`#00FF00`（绿色） |
| `border_color` | STRING | 否 | `#000000` | 十六进制颜色码 | 字体描边颜色。建议用深色以增强可读性 |
| `border_size` | INT | 否 | `2` | 0-20 | 描边宽度（像素）。0=无描边，建议2-4 |
| `shadow_color` | STRING | 否 | `#000000` | 十六进制颜色码 | 阴影颜色。建议用黑色半透明 |
| `shadow_size` | INT | 否 | `2` | 0-20 | 阴影模糊半径（像素）。0=无阴影，建议2-5 |
| `shadow_offset_x` | INT | 否 | `2` | -20~20 | 阴影水平偏移（像素）。正数向右偏移，负数向左 |
| `shadow_offset_y` | INT | 否 | `2` | -20~20 | 阴影垂直偏移（像素）。正数向下偏移，负数向上 |
| `effect` | COMBO | 否 | `fade` | `fade`/`none` | 字幕出现/消失特效。`fade`=淡入淡出，`none`=无特效 |
| `fade_in_duration` | INT | 否 | `300` | 0-2000 | 淡入时长（毫秒）。0=无淡入，直接显示 |
| `fade_out_duration` | INT | 否 | `300` | 0-2000 | 淡出时长（毫秒）。0=无淡出，直接消失 |
| `subtitle_y_position` | FLOAT | 否 | `0.70` | 0.0-1.0 | 字幕垂直位置。0.0=最顶部，1.0=最底部，0.70=底部偏上（推荐） |
| `subtitle_x_margin` | FLOAT | 否 | `0.20` | 0.0-0.5 | 水平总留白比例。0.20=左右各留10%空白。控制长文字自动折行的可用宽度 |
| `fps` | FLOAT | 否 | `24.0` | 1.0-120.0 | 视频帧率。必须与实际视频帧率一致，否则字幕时间会对不上 |

### 输出参数

| 输出 | 类型 | 格式 | 作用说明 |
|------|------|------|----------|
| `images_with_subtitle` | IMAGE | `(B, H, W, C)` 张量 | 合成字幕后的视频帧。可传入 SaveImage、VideoCombine 等节点保存或预览 |

### 字体配置

#### 内置字体

- 插件自带 **阿里巴巴普惠体 Medium**（免费商用）
- 首次加载时自动下载到 `fonts/` 目录（约 6.3MB）

#### 添加自定义字体

1. 将 `.ttf`/`.otf`/`.ttc` 字体文件放入以下任一目录：
   - `ComfyUI/custom_nodes/ComfyUI-Audio-Srt-Aligner/fonts/`
   - `ComfyUI/models/fonts/`
2. 重启 ComfyUI 或在节点上刷新
3. 字体将自动出现在下拉菜单中

### 自动折行机制

当单行字幕文字宽度超过可用区域时，会自动折行为多行：

- **可用宽度计算**：`可用宽度 = 图片宽度 × (1 - subtitle_x_margin)`
- **折行触发条件**：如 `subtitle_x_margin=0.20`，图片宽1280px，则可用宽度为 1024px（左右各留128px）
- **折行方式**：按字符逐个测量宽度，超出时换行
- **多行对齐**：每行独立水平居中

**示例**（字幕文字很长）：
```
第一行文字居中显示在这里
第二行文字居中显示在这里
```

### 特效说明

#### 淡入淡出（fade）

- **淡入**：字幕开始显示时，透明度从 0% → 100% 渐变
- **淡出**：字幕即将消失时，透明度从 100% → 0% 渐变
- **持续时间**：由 `fade_in_duration` 和 `fade_out_duration` 控制（毫秒）
- **效果**：字幕出现和消失更平滑，不突兀

#### 无特效（none）

- 字幕直接显示/消失，无透明度变化
- 适合快速预览或特效由后期处理的情况

---

## 依赖

- `faster-whisper>=1.1.0` — Whisper 语音识别引擎
- `numpy` — 数值计算
- `Pillow>=10.0.0` — 图像处理与字幕渲染
- `av>=10.0.0` — 音频解码
- `jieba` — 中文分词（用于字幕智能切分）

---

## 注意事项

### Whisper 模型

- 首次运行时会自动下载指定的 Whisper 模型（下载到 `ComfyUI/models/stt/whisper/`）
- 模型大小：tiny(~39MB)、base(~74MB)、small(~244MB)、medium(~766MB)、large-v3(~1.5GB)
- 中国大陆用户下载缓慢时，插件会自动使用 `hf-mirror.com` 镜像
- **Apple Silicon (M1/M2/M3)**：`float16` 计算类型可能不可用，建议用 `int8`

### 语言设置

- 中文音频建议明确指定 `language=zh`，自动检测可能误识别为其他语言
- 多语言混合音频建议分段处理或分别指定语言

### 字体与字幕显示

- 中文字幕必须选择支持中文的字体（如阿里巴巴普惠体），否则会显示为方框或乱码
- 字体大小建议 48-120 像素，过小看不清，过大遮挡画面
- 建议同时开启描边（border）和阴影（shadow）以增强可读性

### 性能优化

- `max_chars` 设置过小会导致字幕条目过多，建议 10-15
- 视频分辨率越高，渲染越慢，建议预览时用低分辨率
- 如不需要淡入淡出特效，设置为 `none` 可加快渲染速度

---

## 完整示例工作流

1. **准备音频/视频**：使用 LoadAudio 或 LoadVideo 节点加载素材
2. **生成字幕**：AudioSrtAligner 节点处理音频，输出 `srt_string`
3. **（可选）校对**：如有参考文本，填入进行校对；否则留空
4. **合成字幕**：VideoSrtOverlay 节点将字幕合成到视频帧
5. **导出**：使用 VideoCombine 或 SaveImage 节点保存结果

详细工作流文件：[Audio-srt-aligner--srt-overlay.json](example/Audio-srt-aligner--srt-overlay.json)
