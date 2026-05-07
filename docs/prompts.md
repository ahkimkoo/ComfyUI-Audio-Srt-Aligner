# ComfyUI Plugin: Audio SRT Aligner — 开发提示词

> 请严格按照以下要求开发 ComfyUI 自定义节点插件。

---

## 一、项目背景

### 1.1 核心项目
`audio-srt-aligner`（源码位于 `/Users/cherokee/Project/audio-srt-aligner`）是一个基于 faster-whisper 的**文稿校对生成字幕**工具。

**工作原理**：
1. 用户提供一段音频文件和一个参考文本（文稿/台词）。
2. 使用 faster-whisper 对音频进行 ASR 识别，获取带时间戳的词级 token。
3. 通过 difflib 序列比对，将参考文本与 ASR token 对齐。
4. 为参考文本中的每个 token 推断时间戳，并组合成字幕条目。
5. 利用波形 VAD 进行边界微调，输出标准的 `.srt` 字幕文件。

**核心优势**：相比纯 ASR 生成字幕，文稿校对方式的字幕文本 100% 准确（与参考文本一致），时间戳由 ASR 驱动，精度极高。实测对齐率可达 97%+。

**关键文件**：`align_to_srt.py`（1519 行），包含：
- `AlignmentConfig` — 配置数据类
- `AlignmentRunResult` — 运行结果数据类
- `transcribe_to_tokens()` — Whisper ASR + token 提取
- `run_alignment_pipeline()` — 完整对齐流程入口函数
- `write_srt()` — SRT 文件写入

### 1.2 ComfyUI 环境
- **conda 环境路径**：`/opt/homebrew/anaconda3/envs/comfyui` (Python 3.10)
- **ComfyUI 安装路径**：`~/App/ComfyUI/`
- **自定义节点目录**：`~/App/ComfyUI/custom_nodes/`
- **输出目录**：`~/App/ComfyUI/output/`
- **输入目录**：`~/App/ComfyUI/input/`

---

## 二、需求说明

### 2.1 功能目标
在 ComfyUI 中创建一个节点，接收**音频文件路径**和**参考文本**作为输入，调用 `audio-srt-aligner` 的对齐逻辑生成 SRT 字幕字符串，并**直接返回 SRT 格式的字符串**（不写文件），方便下游节点直接使用。

### 2.2 技术要求

#### 2.2.1 项目结构
```
ComfyUI-Audio-Srt-Aligner/
├── __init__.py              # ComfyUI 入口，节点注册，首次运行自动下载字体
├── nodes/
│   ├── __init__.py          # 空文件
│   └── aligner_node.py      # 核心节点实现
├── aligner/                  # 从 audio-srt-aligner 移植的对齐逻辑
│   ├── __init__.py
│   └── engine.py             # 封装 AlignmentConfig + run_alignment_pipeline
├── scripts/
│   └── download_fonts.py     # 字体下载脚本（首次运行时调用）
├── requirements.txt         # 依赖列表
├── .gitignore
└── README.md
```

**模型与字体目录**（由 ComfyUI 管理，不在插件仓库内）：
- `comfyui/models/stt/whisper/` — Whisper 模型（`small.pt`、`large-v3.pt` 等）
- `comfyui/models/fonts/` — 字幕字体文件（`NotoSansSC-Regular.otf` 等）

#### 2.2.2 __init__.py 关键注意事项（CRITICAL）
`nodes/` 子目录会与 ComfyUI 自带的 `nodes.py` 模块命名冲突！必须使用 `importlib.util` 动态加载：

```python
import os
import sys
import importlib.util

plugin_dir = os.path.dirname(os.path.abspath(__file__))

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# 使用唯一模块名前缀避免冲突
_aligner_node = _load_module(
    "audio_srt_aligner_node",
    os.path.join(plugin_dir, "nodes", "aligner_node.py")
)

NODE_CLASS_MAPPINGS = {
    "AudioSrtAligner": _aligner_node.AudioSrtAligner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSrtAligner": "Audio SRT Aligner (文稿校对字幕)",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

**路径配置**（在 `__init__.py` 中定义）：
```python
import folder_paths
# ComfyUI 内置的路径管理
COMFYUI_MODELS = folder_paths.get_folder_paths("custom_nodes")  # 获取 ComfyUI 根路径推断
WHISPER_MODEL_DIR = os.path.join(folder_paths.models_dir, "stt", "whisper")
FONTS_DIR = os.path.join(folder_paths.models_dir, "fonts")

# 确保目录存在
os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)
```

如果 `folder_paths` 不可用，使用环境变量 `COMFYUI_PATH` 或 ComfyUI 安装路径推导。

#### 2.2.3 依赖
`requirements.txt`：
```
faster-whisper>=1.1.0
numpy
```
注意：`av`（PyAV）是可选依赖，如果引擎中使用到音频读取则需要添加。

#### 2.2.4 Whisper 模型管理
- **模型存放路径**：Whisper 模型统一存放在 ComfyUI 的 `models/stt/whisper/` 目录下，例如 `large-v3.pt`、`medium.pt`、`small.pt` 等。
- **模型检测与自动下载**：节点运行时必须先检测目标模型是否已存在于 `models/stt/whisper/` 目录下。如果模型不存在，则自动从 HuggingFace 下载到该目录；如果已存在则直接加载，避免重复下载。
- **中国大陆网络问题**：下载可能超时失败。需要在代码中设置代理（`HF_ENDPOINT=https://hf-mirror.com` + HTTP/SOCKS代理）或提供明确的错误提示。
- 默认使用 `small` 模型（平衡速度与精度），对中文效果良好。用户可通过 `model_size` 参数选择 `tiny`、`base`、`small`、`medium`、`large-v3`。

#### 2.2.5 对齐逻辑移植策略
**不要**直接 import 原项目的 `align_to_srt.py`（它是 1500+ 行的脚本文件，包含 CLI 解析）。

**正确做法**：
1. 将核心函数（`transcribe_to_tokens`, `run_alignment_pipeline`, `AlignmentConfig` 等）复制到 `aligner/engine.py` 中。
2. 移除 `argparse` 相关代码，改用函数参数传递配置。
3. 保留所有数据类和对齐算法逻辑。
4. 添加 `progress` 回调支持，用于 ComfyUI 节点进度显示（通过 `pbar` 或日志）。

---

## 三、ComfyUI 节点规格

### 3.1 节点定义

```python
class AudioSrtAligner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/audio.wav",
                    "multiline": False,
                }),
                "reference_text": ("STRING", {
                    "default": "",
                    "placeholder": "在这里输入参考文本/台词文稿...",
                    "multiline": True,
                }),
                "model_size": (["tiny", "base", "small", "medium", "large-v3"], {
                    "default": "small",
                }),
                "language": ("STRING", {
                    "default": "zh",
                    "placeholder": "zh, en, ja... 留空自动检测",
                    "multiline": False,
                }),
            },
            "optional": {
                "beam_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "compute_type": (["int8", "int8_float16", "float16", "float32"], {
                    "default": "int8",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("srt_string", "detected_language", "srt_entries", "coverage")
    FUNCTION = "process"
    CATEGORY = "audio/srt"

    def process(self, audio_path, reference_text, model_size="small",
                language="zh", beam_size=5, compute_type="int8"):
        """
        执行文稿校对对齐，返回 SRT 格式字符串。
        返回: (srt_string, detected_language, srt_entries, coverage)
        """
        # 实现对齐逻辑
        ...
```

### 3.2 输入说明
| 输入名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `audio_path` | STRING | 是 | 音频文件绝对路径（wav/mp3/aiff 等） |
| `reference_text` | STRING | 是 | 参考文本/台词文稿，多行文本框 |
| `model_size` | ENUM | 是 | Whisper 模型大小，默认 `small` |
| `language` | STRING | 是 | 语言代码，默认 `zh`，留空则自动检测 |
| `beam_size` | INT | 否 | 搜索束宽，默认 5 |
| `compute_type` | ENUM | 否 | 计算精度，根据实际硬件选择（CUDA可用推荐 `float16`，CPU推荐 `int8`） |

### 3.3 输出说明
| 输出名 | 类型 | 说明 |
|--------|------|------|
| `srt_string` | STRING | 生成的 SRT 格式字幕字符串（可直接连接到下游文本节点） |
| `detected_language` | STRING | Whisper 检测到的语言代码 |
| `srt_entries` | INT | 生成的 SRT 条目数量 |
| `coverage` | FLOAT | 文本对齐覆盖率（0.0 ~ 1.0） |

### 3.4 内部处理流程
1. **验证输入**：检查音频文件是否存在，参考文本非空。
2. **保存临时文本**：将 `reference_text` 写入临时文件（对齐引擎需要文件路径输入）。
3. **初始化引擎**：创建 `AlignmentConfig`，设置模型、设备、语言等参数。
4. **运行对齐**：调用 `run_alignment_pipeline()`。
5. **生成 SRT 字符串**：将结果转为标准 SRT 格式字符串（不要写文件，直接返回字符串）。
6. **返回结果**：输出 SRT 字符串、检测语言、条目数、覆盖率。
7. **清理临时文件**：删除临时文本文件。

### 3.5 SRT 格式约定
输出为标准 SRT 格式字符串，例如：
```
1
00:00:00,000 --> 00:00:03,500
你好，这是第一段测试。

2
00:00:03,500 --> 00:00:07,200
用于验证字幕对齐功能。
```

### 3.6 错误处理
- 音频文件不存在 → 抛出 `ValueError`，消息包含路径。
- Whisper 模型下载失败 → 捕获异常，提示用户检查网络或手动下载。
- 对齐失败（空文本、无有效 token）→ 抛出 `ValueError`，包含原因。
- 所有错误通过 `raise Exception(...)` 抛出，ComfyUI 会在界面上显示红色错误信息。

---

## 四、开发完成后本地测试要求

### 4.1 安装依赖
```bash
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate comfyui
cd /Users/cherokee/Project/ComfyUI-Audio-Srt-Aligner
pip install -r requirements.txt
```

### 4.2 创建符号链接到 ComfyUI
```bash
ln -sf /Users/cherokee/Project/ComfyUI-Audio-Srt-Aligner \
       ~/App/ComfyUI/custom_nodes/ComfyUI-Audio-Srt-Aligner
```

### 4.3 启动 ComfyUI 验证节点加载
```bash
cd ~/App/ComfyUI
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate comfyui
python main.py --cpu --listen 127.0.0.1 --port 8188 &
```
等待启动完成后，检查日志中是否有 `Audio SRT Aligner` 节点加载成功的信息：
```bash
# 查看日志（ComfyUI 输出到 stderr）
cat ~/App/ComfyUI/user/comfyui_8188.log | grep -i "audio.*srt\|aligner\|IMPORT"
```

### 4.4 API 工作流测试
使用 ComfyUI API 提交一个测试工作流，验证节点能否正常执行：

```python
# test_api.py — 使用 /opt/homebrew/anaconda3/envs/comfyui/bin/python 运行
import json
import urllib.request
import time

# 准备测试音频和文本
AUDIO_PATH = "/Users/cherokee/Project/audio-srt-aligner/aigc_audio.wav"
REFERENCE_TEXT = "你好，这是一段测试音频，用于验证 ComfyUI 插件的字幕对齐功能。"

workflow = {
    "prompt": {
        "1": {
            "class_type": "AudioSrtAligner",
            "inputs": {
                "audio_path": AUDIO_PATH,
                "reference_text": REFERENCE_TEXT,
                "model_size": "tiny",  # 使用 tiny 模型加速测试
                "language": "zh",
            }
        },
        "2": {
            "class_type": "ShowText|pysssss",  # 或任何文本预览节点
            "inputs": {
                "text": ["1", 0],  # 连接 AudioSrtAligner 的 srt_string 输出
            }
        }
    }
}

data = json.dumps(workflow).encode('utf-8')
req = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=data,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req, timeout=30) as resp:
    result = json.loads(resp.read())
    prompt_id = result['prompt_id']
    print(f"Submitted prompt_id: {prompt_id}")

# 轮询结果
for i in range(120):  # 最多等待 4 分钟
    time.sleep(2)
    with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}") as resp:
        history = json.loads(resp.read())
    if prompt_id in history:
        output = history[prompt_id].get('outputs', {})
        status = history[prompt_id].get('status', {})
        if status.get('completed'):
            print(f"SUCCESS: {json.dumps(output, indent=2, ensure_ascii=False)}")
            break
        elif status.get('status_str') == 'error':
            print(f"ERROR: {status.get('messages')}")
            break
else:
    print("TIMEOUT: Workflow did not complete within 4 minutes")
```

### 4.5 验证输出
- 检查 API 返回的 `outputs` 中节点 1 的第一个输出是否为有效的 SRT 格式字符串。
- 确认字符串包含正确的 SRT 格式（序号、时间戳、文本）。
- 确认覆盖率（coverage）> 80%。

### 4.6 清理
测试完成后，停止 ComfyUI 进程：
```bash
pkill -f "python main.py"
```

---

## 五、注意事项

1. **运行环境**：本项目作为 ComfyUI 自定义节点运行，ComfyUI 运行在什么环境上，本节点就运行在什么环境上。支持 GPU（CUDA/ROCRun/MPS）和 CPU 环境。`compute_type` 应根据实际硬件选择：CUDA 环境推荐 `float16`，CPU 环境推荐 `int8`，设备参数应使用 `device="auto"` 让 faster-whisper 自动检测可用设备。
2. **模型缓存**：faster-whisper 默认将模型缓存到 `~/.cache/huggingface/`。首次运行下载较慢，后续运行会复用缓存。
3. **中文支持**：`language="zh"` 时 Whisper 对中文识别效果最好。如果用户留空，Whisper 会自动检测语言。
4. **大文件处理**：对于超长音频（> 30 分钟），Whisper 处理可能很慢。建议在 README 中说明推荐使用 `small` 或 `base` 模型。
5. **Git 忽略**：在 `.gitignore` 中排除 `__pycache__/`、`*.pyc`、`*.pt`（模型文件）、`output/`。

---

## 六、第二阶段：视频合成字幕节点

> 在第一阶段完成后，继续开发第二个节点：**将 SRT 字幕合成到图像/视频帧上**。

### 6.1 功能目标

创建一个节点，接收 **IMAGE(batch)** 和 **SRT 格式字符串**，根据 SRT 时间戳将字幕渲染到每一帧图像上。字幕**水平居中**，支持丰富的样式参数（字体、颜色、阴影、淡入淡出等），输出叠加字幕后的 IMAGE(batch)。

### 6.2 节点定义

```python
class VideoSrtOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "srt_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "font_family": ("STRING", {
                    "default": "",
                    "placeholder": "留空使用系统默认字体，或指定字体文件路径",
                    "multiline": False,
                }),
                "font_size": ("INT", {
                    "default": 48,
                    "min": 12,
                    "max": 200,
                    "step": 1,
                }),
                "font_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                }),
                "border_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                }),
                "border_size": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                }),
                "shadow_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                }),
                "shadow_size": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                }),
                "shadow_offset_x": ("INT", {
                    "default": 2,
                    "min": -20,
                    "max": 20,
                    "step": 1,
                }),
                "shadow_offset_y": ("INT", {
                    "default": 2,
                    "min": -20,
                    "max": 20,
                    "step": 1,
                }),
                "effect": (["fade", "none"], {
                    "default": "fade",
                }),
                "fade_in_duration": ("INT", {
                    "default": 300,
                    "min": 0,
                    "max": 2000,
                    "step": 50,
                }),
                "fade_out_duration": ("INT", {
                    "default": 300,
                    "min": 0,
                    "max": 2000,
                    "step": 50,
                }),
            },
            "optional": {
                "subtitle_y_position": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                }),
                "subtitle_x_margin": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images_with_subtitle",)
    FUNCTION = "process"
    CATEGORY = "video/srt"

    def process(self, images, srt_string, font_family, font_size,
                font_color, border_color, border_size,
                shadow_color, shadow_size, shadow_offset_x, shadow_offset_y,
                effect, fade_in_duration, fade_out_duration,
                subtitle_y_position=0.3, subtitle_x_margin=0.3, fps=24.0):
        """
        将 SRT 字幕叠加到图像 batch 上。
        返回: 叠加字幕后的 IMAGE tensor
        """
        ...
```

### 6.3 输入说明

| 输入名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `images` | IMAGE | 是 | - | 输入图像 batch (B, H, W, C) |
| `srt_string` | STRING | 是 | - | SRT 格式字幕字符串 |
| `font_family` | STRING | 是 | 见下方说明 | 字体文件路径（需根据操作系统选择，见注意事项） |
| `font_size` | INT | 是 | 48 | 字幕字号 (12-200) |
| `font_color` | STRING | 是 | `#FFFFFF` | 字体颜色 (HEX) |
| `border_color` | STRING | 是 | `#000000` | 边框颜色 (HEX) |
| `border_size` | INT | 是 | 2 | 边框大小 (0-20) |
| `shadow_color` | STRING | 是 | `#000000` | 阴影颜色 (HEX) |
| `shadow_size` | INT | 是 | 3 | 阴影大小/模糊半径 (0-20) |
| `shadow_offset_x` | INT | 是 | 2 | 阴影 X 偏移 (-20~20) |
| `shadow_offset_y` | INT | 是 | 2 | 阴影 Y 偏移 (-20~20) |
| `effect` | ENUM | 是 | `fade` | 字幕效果：`fade`（淡入淡出）/ `none`（无效果） |
| `fade_in_duration` | INT | 是 | 300 | 入场时间（毫秒，0-2000） |
| `fade_out_duration` | INT | 是 | 300 | 出场时间（毫秒，0-2000） |
| `subtitle_y_position` | FLOAT | 否 | 0.3 | 字幕上下位置（屏幕高度的百分比，0.0=顶部，1.0=底部） |
| `subtitle_x_margin` | FLOAT | 否 | 0.3 | 字幕左右边距（屏幕宽度的百分比，0.0=无边距，0.5=各留50%） |
| `fps` | FLOAT | 否 | 24.0 | 帧率，用于将 SRT 时间戳映射到帧索引 |

### 6.4 输出说明

| 输出名 | 类型 | 说明 |
|--------|------|------|
| `images_with_subtitle` | IMAGE | 叠加字幕后的图像 batch，形状与输入一致 |

### 6.5 内部处理流程

1. **解析 SRT**：将 SRT 字符串解析为条目列表，每个条目包含序号、开始时间（秒）、结束时间（秒）、文本内容。
2. **计算时间映射**：根据 `fps` 和图像 batch 的帧数，确定每一帧对应的时间点（`frame_index / fps`）。
3. **遍历每一帧**：
   - 找到当前时间点应该显示的所有 SRT 条目（可能有多行字幕同时显示）。
   - 合并为单行文本（多条目之间用空格或换行分隔）。
   - 根据 `effect` 参数计算透明度：
     - `fade`：在入场时间段内从 0 渐变到 1，在出场时间段内从 1 渐变到 0，中间保持 1。
     - `none`：始终不透明度为 1。
4. **渲染字幕到图像**：
   - 使用 Pillow 或 OpenCV 在图像上绘制文本。
   - **水平居中**：根据文本宽度和 `subtitle_x_margin` 计算 X 坐标，确保字幕在可用宽度内居中。
   - **垂直位置**：根据 `subtitle_y_position` 计算 Y 坐标。
   - 应用字体、字号、颜色、边框、阴影等样式。
   - 根据透明度混合字幕层和原始图像层。
5. **批量处理**：使用 numpy/torch 向量化操作或逐帧处理后堆叠。
6. **返回结果**：输出叠加字幕后的 IMAGE tensor。

### 6.6 SRT 解析规则

```python
def parse_srt(srt_string):
    """
    解析 SRT 字符串，返回条目列表。
    每个条目：{
        'index': int,
        'start': float,  # 秒
        'end': float,    # 秒
        'text': str
    }
    """
    ...
```

- 时间格式：`HH:MM:SS,mmm --> HH:MM:SS,mmm`
- 忽略空行
- 支持多行文本（合并为一行，用空格连接）

### 6.7 字幕渲染实现要点

1. **Pillow 方案**（推荐，跨平台兼容性好）：
   - 使用 `ImageDraw.text()` 或 `ImageDraw.textbbox()` 计算文本边界框。
   - 使用 `ImageFont.truetype()` 加载字体。
   - 边框和阴影通过多次绘制实现（偏移绘制描边）。
   - 透明度通过 `Image.blend()` 或手动 alpha 混合实现。

2. **OpenCV 方案**（备选）：
   - 使用 `cv2.putText()` 绘制文本。
   - 中文字体支持较弱，需要额外处理。
   - 阴影和边框实现较复杂。

3. **字体管理**：
   - 字体文件统一存放在 ComfyUI 的 `models/fonts/` 目录下。
   - **开发要求**：在项目的安装脚本或 `__init__.py` 首次运行时，自动从网上下载常见字幕字体到 `models/fonts/`，至少包含：
     - **Noto Sans SC**（思源黑体，Google Fonts，开源中文字体，字幕效果好）
     - 其他常用开源中文字体
   - 节点运行时从 `models/fonts/` 目录下加载字体。`font_family` 参数可以指定文件名（如 `NotoSansSC-Regular.otf`）或留空使用默认字体。
   - 建议字体下拉选项列出 `models/fonts/` 下所有可用字体。
   - `.ttc` 字体集合可能需要指定索引，或使用 `PIL.ImageFont` 的字体索引参数。

4. **性能优化**：
   - 对于相同文本的连续帧，可以缓存渲染结果。
   - 使用 numpy 向量化混合操作替代逐像素 Python 循环。

### 6.8 淡入淡出效果实现

```python
def calculate_opacity(current_time, start_time, end_time,
                      fade_in_ms, fade_out_ms):
    """
    计算当前帧的字幕透明度。
    fade_in_ms / fade_out_ms 为毫秒。
    """
    fade_in_sec = fade_in_ms / 1000.0
    fade_out_sec = fade_out_ms / 1000.0

    if current_time < start_time + fade_in_sec:
        # 入场阶段：从 0 渐变到 1
        return max(0.0, (current_time - start_time) / fade_in_sec)
    elif current_time > end_time - fade_out_sec:
        # 出场阶段：从 1 渐变到 0
        return max(0.0, (end_time - current_time) / fade_out_sec)
    else:
        # 中间阶段：完全不透明
        return 1.0
```

### 6.9 字幕位置计算

```python
def calculate_position(image_width, image_height, text_width,
                       y_position, x_margin):
    """
    计算字幕居中位置。
    y_position: 0.0-1.0，字幕中心在屏幕高度的百分比位置
    x_margin: 0.0-0.5，左右边距占屏幕宽度的百分比
    """
    # 水平居中：在可用宽度内居中
    available_width = image_width * (1.0 - 2 * x_margin)
    x = x_margin * image_width + (available_width - text_width) / 2

    # 垂直位置：y_position 是字幕中心在屏幕高度的百分比
    y = y_position * image_height

    return x, y
```

### 6.10 错误处理

- SRT 格式错误 → 抛出 `ValueError`，包含具体错误行号。
- 字体文件不存在 → 抛出 `ValueError`，提示检查路径。
- 图像 batch 为空 → 抛出 `ValueError`。
- 无有效字幕条目 → 返回原始图像，不报错。

### 6.11 __init__.py 节点注册更新

在 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 中添加新节点：

```python
_video_srt_node = _load_module(
    "video_srt_overlay_node",
    os.path.join(plugin_dir, "nodes", "video_srt_overlay_node.py")
)

NODE_CLASS_MAPPINGS = {
    "AudioSrtAligner": _aligner_node.AudioSrtAligner,
    "VideoSrtOverlay": _video_srt_node.VideoSrtOverlay,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSrtAligner": "Audio SRT Aligner (文稿校对字幕)",
    "VideoSrtOverlay": "Video SRT Overlay (字幕合成)",
}
```

### 6.12 项目结构更新

```
ComfyUI-Audio-Srt-Aligner/
├── __init__.py              # 入口文件，注册两个节点
├── nodes/
│   ├── __init__.py          # 空文件
│   ├── aligner_node.py      # 第一阶段：文稿校对对齐节点
│   └── video_srt_overlay_node.py  # 第二阶段：字幕合成节点
├── aligner/
│   ├── __init__.py
│   └── engine.py             # 对齐引擎
├── utils/
│   ├── __init__.py
│   └── srt_parser.py         # SRT 解析工具
├── requirements.txt         # 依赖列表（新增 Pillow）
├── .gitignore
└── README.md
```

`requirements.txt` 新增：
```
Pillow>=10.0.0
```

---

## 七、开发完成后本地测试要求

### 7.1 安装依赖

```bash
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate comfyui
cd /Users/cherokee/Project/ComfyUI-Audio-Srt-Aligner
pip install -r requirements.txt
```

### 7.2 创建符号链接到 ComfyUI

```bash
ln -sf /Users/cherokee/Project/ComfyUI-Audio-Srt-Aligner \
       ~/App/ComfyUI/custom_nodes/ComfyUI-Audio-Srt-Aligner
```

### 7.3 启动 ComfyUI 验证节点加载

```bash
cd ~/App/ComfyUI
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate comfyui
python main.py --cpu --listen 127.0.0.1 --port 8188 &
```

等待启动完成后，检查日志：

```bash
cat ~/App/ComfyUI/user/comfyui_8188.log | grep -i "audio.*srt\|video.*srt\|aligner\|IMPORT"
```

### 7.4 API 工作流测试（两阶段完整测试）

```python
# test_api_full.py — 使用 /opt/homebrew/anaconda3/envs/comfyui/bin/python 运行
import json
import urllib.request
import time

AUDIO_PATH = "/Users/cherokee/Project/audio-srt-aligner/aigc_audio.wav"
REFERENCE_TEXT = "你好，这是一段测试音频，用于验证 ComfyUI 插件的字幕对齐功能。"

workflow = {
    "prompt": {
        "1": {
            "class_type": "AudioSrtAligner",
            "inputs": {
                "audio_path": AUDIO_PATH,
                "reference_text": REFERENCE_TEXT,
                "model_size": "tiny",
                "language": "zh",
            }
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "test_frame.png",  # 准备一张测试图片
            }
        },
        "3": {
            "class_type": "BatchImage",
            "inputs": {
                "images": [["2", 0]] * 50,  # 复制 50 帧模拟视频
            }
        },
        "4": {
            "class_type": "VideoSrtOverlay",
            "inputs": {
                "images": ["3", 0],
                "srt_string": ["1", 0],
                "font_size": 48,
                "font_color": "#FFFFFF",
                "effect": "fade",
                "fade_in_duration": 300,
                "fade_out_duration": 300,
                "fps": 24.0,
            }
        }
    }
}

data = json.dumps(workflow).encode('utf-8')
req = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=data,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req, timeout=30) as resp:
    result = json.loads(resp.read())
    prompt_id = result['prompt_id']
    print(f"Submitted prompt_id: {prompt_id}")

# 轮询结果
for i in range(120):
    time.sleep(2)
    with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}") as resp:
        history = json.loads(resp.read())
    if prompt_id in history:
        output = history[prompt_id].get('outputs', {})
        status = history[prompt_id].get('status', {})
        if status.get('completed'):
            print(f"SUCCESS: {json.dumps(output, indent=2, ensure_ascii=False)}")
            break
        elif status.get('status_str') == 'error':
            print(f"ERROR: {status.get('messages')}")
            break
else:
    print("TIMEOUT: Workflow did not complete within 4 minutes")
```

### 7.5 验证输出

- 确认 `VideoSrtOverlay` 节点成功执行。
- 检查输出的图像 batch 尺寸与输入一致。
- 抽样检查几帧图像，确认字幕正确叠加在画面上。
- 验证淡入淡出效果（透明度渐变）。
- 确认字幕水平居中，垂直位置符合预期。

### 7.6 清理

测试完成后，停止 ComfyUI 进程：

```bash
pkill -f "python main.py"
```

---

## 八、注意事项

1. **运行环境**：本项目作为 ComfyUI 自定义节点运行，ComfyUI 运行在什么环境上，本节点就运行在什么环境上。支持 GPU（CUDA/ROCRun/MPS）和 CPU 环境。`compute_type` 应根据实际硬件选择：CUDA 环境推荐 `float16`，CPU 环境推荐 `int8`，设备参数应使用 `device="auto"` 让 faster-whisper 自动检测可用设备。
2. **模型缓存**：faster-whisper 默认将模型缓存到 `~/.cache/huggingface/`。首次运行下载较慢，后续运行会复用缓存。
3. **中文支持**：`language="zh"` 时 Whisper 对中文识别效果最好。如果用户留空，Whisper 会自动检测语言。
4. **大文件处理**：对于超长音频（> 30 分钟），Whisper 处理可能很慢。建议在 README 中说明推荐使用 `small` 或 `base` 模型。
5. **Git 忽略**：在 `.gitignore` 中排除 `__pycache__/`、`*.pyc`、`*.pt`（模型文件）、`output/`。
6. **字体渲染**：Pillow 的 `textbbox()` 在某些字体上可能返回不准确的边界框，需要测试并调整。
7. **性能**：对于大尺寸图像（4K），逐帧渲染可能较慢。建议添加进度回调或分块处理。
8. **字幕换行**：当前版本不支持自动换行，如果文本超过可用宽度，可能需要截断或缩小字号。后续版本可添加自动换行支持。

---

## 九、交付物清单

### 第一阶段
- [ ] `__init__.py` — 入口文件，节点注册（含 importlib 冲突处理）
- [ ] `nodes/aligner_node.py` — ComfyUI 节点实现（输出 SRT 字符串，非文件路径）
- [ ] `aligner/engine.py` — 对齐引擎（从 align_to_srt.py 移植的核心逻辑）
- [ ] `aligner/__init__.py` — 空文件
- [ ] `nodes/__init__.py` — 空文件
- [ ] `requirements.txt` — 依赖列表
- [ ] `.gitignore` — Git 忽略规则
- [ ] `README.md` — 使用说明
- [ ] `test_api.py` — API 测试脚本
- [ ] 本地 ComfyUI 测试通过（节点加载成功 + 工作流执行成功 + SRT 字符串输出正确）

### 第二阶段
- [ ] `nodes/video_srt_overlay_node.py` — 字幕合成节点实现
- [ ] `utils/srt_parser.py` — SRT 解析工具
- [ ] `utils/__init__.py` — 空文件
- [ ] `requirements.txt` 更新（添加 Pillow）
- [ ] `__init__.py` 更新（注册第二个节点）
- [ ] `README.md` 更新（添加字幕合成节点使用说明）
- [ ] `test_api_full.py` — 两阶段完整测试脚本
- [ ] 本地 ComfyUI 测试通过（节点加载成功 + 工作流执行成功 + 字幕叠加效果正确）
- [ ] 验证淡入淡出效果、位置、样式参数均正常工作
