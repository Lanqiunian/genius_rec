import sys
import torch
import pathlib
from pathlib import PureWindowsPath

# 1. 自定义一个 WindowsPathMock 继承自 PureWindowsPath（无需再引用 _windows_flavour）
class WindowsPathMock(PureWindowsPath):
    pass

# 将 pathlib.WindowsPath 替换为我们的 mock 类
sys.modules['pathlib'].WindowsPath = WindowsPathMock

# 2. 加载原文件
checkpoint = torch.load('/root/autodl-tmp/genius_rec-main/checkpoints/hstu_encoder.pth', map_location='cpu')

# 3. 转换为兼容路径
def convert_path(obj):
    if isinstance(obj, pathlib.WindowsPath):
        return pathlib.PosixPath(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_path(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_path(x) for x in obj]
    return obj

converted_checkpoint = convert_path(checkpoint)

# 4. 保存新的文件
torch.save(converted_checkpoint, '/root/autodl-tmp/genius_rec-main/checkpoints/checkpoint_universal.pth')
print("已成功转换路径并保存为 checkpoint_universal.pth")