#!/usr/bin/env python3
"""
从 resources\\flashswf\\portraits 下的 SWF 中批量导出带帧标签的立绘 PNG 或 WebP，
并生成形如「NPC名#情绪.png」或「NPC名#情绪.webp」的文件到 resources\\flashswf\\portraits\\illustration 目录。
可选 --webp 以 0.85 质量输出 WebP，节省空间且便于发给 AI 大模型。

依赖：JPEXS Free Flash Decompiler（FFDec）命令行工具。

用法示例（Windows）：

1. 如果你下载的是 ffdec-cli.exe / ffdec.exe：
   python scripts/extract_portraits_from_swf.py ^
       --ffdec-path "C:\\Tools\\ffdec-cli.exe"

2. 如果你只有 ffdec.jar：
   python scripts/extract_portraits_from_swf.py ^
       --ffdec-path "C:\\Tools\\ffdec.jar"

脚本逻辑概要：
1. 自动定位外部 resources 目录（与后端运行时逻辑一致）。
2. 扫描 resources\\flashswf\\portraits 目录下的所有 *.swf。
3. 对每个 SWF：
   - 调用 FFDec：-ignorebackground -export frame <临时目录> <swf>
     得到该 SWF 的所有帧 PNG（按时间顺序命名）。
   - 再调用 FFDec：-dumpSWF <swf>，解析输出中的 FrameLabel / ShowFrame 标签，
     建立「帧号 -> 帧标签」映射。
   - 将带标签的帧号对应到导出的 PNG，重命名并复制到
     resources\\flashswf\\portraits\\illustration 下，命名为
     <NPC名称>#<帧标签>.png 或 .webp（使用 --webp 时）。

注意：
- 脚本假设 FrameLabel 标签的顺序与 FFDec 导出的帧顺序一致：
  一般情况下 FrameLabel 出现在对应帧的 ShowFrame 之前，
  因此通过扫描 -dumpSWF 输出时，将标签绑定到「下一个帧号」。
- 如果某个标签对应的帧号超出导出帧数量，会打印警告并跳过。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image, ImageFilter
except ImportError:
    Image = None  # type: ignore[misc, assignment]
    ImageFilter = None  # type: ignore[misc, assignment]


def _get_project_root() -> Path:
    """
    获取项目根目录（cfn-rag-backend），脚本位于 scripts 目录下。
    """
    return Path(__file__).resolve().parent.parent


def _get_resources_dir() -> Path:
    """与后端一致：resources 或 CrazyFlashNight 资源根目录。"""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from services.game_data.paths import find_resources_directory

    return find_resources_directory()


def _build_ffdec_cmd(ffdec_path: Path, extra_args: List[str]) -> List[str]:
    """
    根据 ffdec_path 构造最终的命令行：
    - 如果是 .jar，则使用: java -Dfile.encoding=UTF-8 -jar <jar> <extra_args...>
      （强制 UTF-8 避免 -dumpSWF 输出的中文帧标签在 Windows 下乱码）
    - 否则直接调用可执行文件: <ffdec_path> <extra_args...>
    """
    if ffdec_path.suffix.lower() == ".jar":
        return ["java", "-Dfile.encoding=UTF-8", "-jar", str(ffdec_path), *extra_args]
    return [str(ffdec_path), *extra_args]


def _run_ffdec(ffdec_path: Path, args: List[str]) -> Tuple[int, str, str]:
    """
    运行 FFDec 命令，返回 (exit_code, stdout, stderr)。
    工作目录设为 ffdec 所在目录，以便 java -jar 能通过 manifest 的 Class-Path 找到 lib/*。
    """
    cmd = _build_ffdec_cmd(ffdec_path, args)
    # 必须在 ffdec 所在目录执行，否则 manifest Class-Path 中的 lib/* 无法解析
    cwd = str(ffdec_path.resolve().parent)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


def _process_frame_image(
    src_png: Path,
    dst_path: Path,
    crop_rect: Tuple[int, int, int, int] | None,
    zoom: int,
    smooth: bool,
    save_webp: bool = False,
    webp_quality: float = 0.85,
) -> None:
    """
    将导出的帧 PNG 做可选裁剪与轻度平滑后写入目标路径。
    crop_rect: (left, top, right, bottom) 为原始画布坐标，需乘以 zoom 得到导出图上的像素框。
    save_webp: 为 True 时保存为 WebP（节省空间、便于发给 AI）；否则保存为 PNG。
    """
    if Image is None:
        shutil.copy2(src_png, dst_path)
        return
    img = Image.open(src_png).convert("RGBA")
    if crop_rect is not None:
        left, top, right, bottom = crop_rect
        box = (
            left * zoom,
            top * zoom,
            right * zoom,
            bottom * zoom,
        )
        img = img.crop(box)
    if smooth:
        img = img.filter(ImageFilter.SMOOTH)
    if save_webp:
        quality = max(1, min(100, int(webp_quality * 100)))
        img.save(dst_path, "WEBP", quality=quality)
    else:
        img.save(dst_path, "PNG")


def _frame_png_sort_key(path: Path) -> Tuple[int, str]:
    """
    帧 PNG 的排序键：按文件名中的数字部分数值排序，避免字典序导致 1, 10, 11, 2...
    FFDec 导出名可能为 1.png, 2.png, 10.png 或 frame_1.png 等。
    """
    stem = path.stem
    m = re.search(r"\d+", stem)
    num = int(m.group()) if m else 0
    return (num, stem)


def _export_frames_with_ffdec(
    ffdec_path: Path, swf_path: Path, out_dir: Path, zoom: int = 1
) -> List[Path]:
    """
    使用 FFDec 将单个 SWF 的所有帧导出为 PNG（含矢量内容会被栅格化）。
    zoom: 导出缩放倍数（如 2 即 200%），可提高精细度、减轻锯齿感。
    返回按帧号数值排序的 PNG 列表，与时间顺序一致。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    args: List[str] = []
    if zoom > 1:
        args.extend(["-zoom", str(zoom)])
    args.extend([
        "-ignorebackground",
        "-export",
        "frame",
        str(out_dir.resolve()),
        str(swf_path.resolve()),
    ])
    code, stdout, stderr = _run_ffdec(ffdec_path, args)
    if code != 0:
        raise RuntimeError(
            f"FFDec 导出帧失败，SWF={swf_path}\n"
            f"命令输出:\n{stdout}\n错误输出:\n{stderr}"
        )

    png_files = sorted(out_dir.glob("*.png"), key=_frame_png_sort_key)
    if not png_files:
        print(f"[警告] SWF 中未导出到任何 PNG 帧：{swf_path}")
    else:
        print(f"[信息] SWF={swf_path.name} 导出帧数量: {len(png_files)}")
    return png_files


# FFDec -dumpSWF 中 FrameLabel 的标签格式，兼容 Name:"xxx" / Name: "xxx" / Label: "xxx" 等
_FRAME_LABEL_PATTERN = re.compile(
    r"(?:Name|Label)\s*:\s*\"([^\"]*)\"",
    re.IGNORECASE,
)


def _parse_frame_labels_from_dump(dump_text: str) -> Dict[int, str]:
    """
    从 FFDec 的 -dumpSWF 输出中解析「帧号 -> 帧标签」映射。
    使用正则提取标签，兼容不同版本输出的空格差异（如 Name:"xxx" 与 Name: "xxx"）。
    """
    frame_labels: Dict[int, str] = {}
    current_frame = 0

    for raw_line in dump_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "ShowFrame" in line:
            current_frame += 1
            continue

        if "FrameLabel" in line:
            m = _FRAME_LABEL_PATTERN.search(line)
            if not m:
                continue
            label = m.group(1).strip()
            if not label:
                continue
            frame_index = current_frame + 1
            if frame_index in frame_labels:
                continue
            frame_labels[frame_index] = label

    if frame_labels:
        print(
            f"[信息] 从 dumpSWF 解析到 {len(frame_labels)} 个带标签帧："
            f"{', '.join(f'#{i}:{l}' for i, l in list(frame_labels.items())[:5])}"
            + (" ..." if len(frame_labels) > 5 else "")
        )
    else:
        print("[警告] 未在 dumpSWF 输出中解析到任何 FrameLabel 标签")

    return frame_labels


def _extract_labels_with_ffdec(ffdec_path: Path, swf_path: Path) -> Dict[int, str]:
    """
    调用 FFDec 的 -dumpSWF 获取标签信息，并解析出「帧号 -> 帧标签」。
    """
    code, stdout, stderr = _run_ffdec(ffdec_path, ["-dumpSWF", str(swf_path.resolve())])
    if code != 0:
        raise RuntimeError(
            f"FFDec -dumpSWF 失败，SWF={swf_path}\n"
            f"命令输出:\n{stdout}\n错误输出:\n{stderr}"
        )
    return _parse_frame_labels_from_dump(stdout)


def process_single_swf(
    ffdec_path: Path,
    swf_path: Path,
    illustration_dir: Path,
    temp_root: Path,
    overwrite: bool = False,
    zoom: int = 1,
    crop_rect: Tuple[int, int, int, int] | None = None,
    smooth: bool = False,
    webp: bool = False,
    webp_quality: float = 0.85,
) -> None:
    """
    处理单个 NPC 的 SWF：
    1. 导出所有帧 PNG（可选 zoom 提高分辨率）
    2. 解析帧标签（若无标签则用第一帧作为「普通」）
    3. 将有标签的帧经可选裁剪/平滑后保存为 <npc_name>#<label>.png 或 .webp
    overwrite=False 时已存在的文件不覆盖，True 时覆盖。
    webp=True 时输出 WebP（默认 quality=0.85），节省空间且便于发给 AI 大模型。
    """
    npc_name = swf_path.stem
    print(f"\n===== 处理 SWF：{swf_path.name}（NPC: {npc_name}）=====")

    temp_dir = temp_root / npc_name
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    png_frames = _export_frames_with_ffdec(ffdec_path, swf_path, temp_dir, zoom=zoom)
    if not png_frames:
        print(f"[警告] 跳过：未导出任何帧，SWF={swf_path}")
        return

    frame_labels = _extract_labels_with_ffdec(ffdec_path, swf_path)
    ext = ".webp" if webp else ".png"
    if not frame_labels:
        # 无帧标签时，将第一帧作为「普通」导出
        frame_labels = {1: "普通"}
        print(f"[信息] 未解析到帧标签，将第一帧导出为「{npc_name}#普通{ext}」")

    illustration_dir.mkdir(parents=True, exist_ok=True)

    # 假设 png_frames 已按时间顺序排序，对应帧号从 1 开始
    copied_count = 0
    for frame_index, label in frame_labels.items():
        idx = frame_index - 1
        if idx < 0 or idx >= len(png_frames):
            print(
                f"[警告] 帧号 {frame_index} 超出导出帧范围（共 {len(png_frames)} 帧），"
                f"标签「{label}」将被忽略。"
            )
            continue

        src_png = png_frames[idx]
        out_name = f"{npc_name}#{label}{ext}"
        dst_path = illustration_dir / out_name

        if dst_path.exists() and not overwrite:
            print(f"[跳过] 已存在，未覆盖：{dst_path.name}")
            continue

        if crop_rect is not None or smooth or webp:
            _process_frame_image(
                src_png, dst_path, crop_rect, zoom, smooth,
                save_webp=webp, webp_quality=webp_quality,
            )
        else:
            shutil.copy2(src_png, dst_path)
        copied_count += 1
        print(f"[OK] 生成立绘：{dst_path}")

    print(
        f"[完成] SWF={swf_path.name}：成功生成 {copied_count} 张带标签立绘 {ext.lstrip('.').upper()}，"
        f"输出目录={illustration_dir}"
    )


def run_extract(
    ffdec_path: Path,
    resources_dir: Path | None = None,
    overwrite: bool = False,
    zoom: int = 4,
    smooth: bool = True,
    crop_rect: Tuple[int, int, int, int] | None = None,
    only_npc: str | None = None,
    webp: bool = False,
    webp_quality: float = 0.85,
) -> Dict[str, object]:
    """
    立绘导出入口，可供 API 或脚本 main 调用。
    返回 {"success": bool, "processed": int, "total": int, "error": str | None}。
    """
    try:
        if resources_dir is None:
            resources_dir = _get_resources_dir()
        resources_dir = resources_dir.resolve()
        if not ffdec_path.exists():
            return {"success": False, "processed": 0, "total": 0, "error": f"FFDec 不存在: {ffdec_path}"}
        if not resources_dir.exists():
            return {"success": False, "processed": 0, "total": 0, "error": f"resources 目录不存在: {resources_dir}"}

        portraits_dir = resources_dir / "flashswf" / "portraits"
        if not portraits_dir.exists():
            return {"success": False, "processed": 0, "total": 0, "error": f"portraits 目录不存在: {portraits_dir}"}

        illustration_dir = portraits_dir / "illustration"
        temp_root = portraits_dir / "_tmp_export_frames"
        temp_root.mkdir(parents=True, exist_ok=True)

        try:
            swf_files = sorted(portraits_dir.glob("*.swf"))
            if only_npc:
                target = only_npc.strip()
                swf_files = [p for p in swf_files if p.stem == target]
                if not swf_files:
                    return {"success": False, "processed": 0, "total": 0, "error": f"未找到 NPC 对应 SWF: {target}"}
            if not swf_files:
                return {"success": True, "processed": 0, "total": 0, "error": None}

            if (smooth or webp) and Image is None:
                return {"success": False, "processed": 0, "total": len(swf_files), "error": "启用 smooth 或 webp 需要安装 Pillow"}

            total_processed = 0
            for swf_path in swf_files:
                try:
                    process_single_swf(
                        ffdec_path, swf_path, illustration_dir, temp_root,
                        overwrite=overwrite,
                        zoom=zoom,
                        crop_rect=crop_rect,
                        smooth=smooth,
                        webp=webp,
                        webp_quality=webp_quality,
                    )
                    total_processed += 1
                except Exception as e:
                    print(f"[错误] 处理 SWF={swf_path.name} 时发生异常：{e}")
                    import traceback
                    traceback.print_exc()

            return {"success": True, "processed": total_processed, "total": len(swf_files), "error": None}
        finally:
            # 脚本执行完毕后删除临时帧导出目录
            if temp_root.exists():
                try:
                    shutil.rmtree(temp_root)
                    print(f"[信息] 已删除临时目录：{temp_root}")
                except Exception as e:
                    print(f"[警告] 删除临时目录失败：{temp_root}，{e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "processed": 0, "total": 0, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从 resources/flashswf/portraits 下的 SWF 批量导出带帧标签的立绘 PNG，"
            "生成 <NPC名>#<情绪标签>.png 到 illustration 目录。"
        )
    )
    parser.add_argument(
        "--ffdec-path",
        required=True,
        help=(
            "FFDec 命令行可执行文件路径，支持 ffdec-cli.exe/ffdec.exe 或 ffdec.jar；"
            "若为 .jar，将通过 `java -jar` 方式调用。"
        ),
    )
    parser.add_argument(
        "--resources-dir",
        type=str,
        default=None,
        help=(
            "可选：手动指定 resources 目录；"
            "默认自动推断（与后端运行逻辑一致，优先使用 CFN_RESOURCES_DIR）。"
        ),
    )
    parser.add_argument(
        "--only-npc",
        type=str,
        default=None,
        help="可选：只处理指定 NPC 名称对应的 SWF（不含扩展名），如：Andy Law",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的立绘 PNG；默认不覆盖已存在的文件。",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=2,
        metavar="N",
        help="导出帧的缩放倍数（如 2 即 200%%），提高精细度、减轻锯齿；默认 2。",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        metavar="L,T,R,B",
        help=(
            "裁剪区域：原始画布坐标 left,top,right,bottom，例如 20,30,900,400。"
            "不传则不裁剪。与 --zoom 配合时，脚本会按缩放后的尺寸正确裁剪。"
        ),
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="对导出图做轻度平滑，减轻位图边缘锯齿（需安装 Pillow）。",
    )
    parser.add_argument(
        "--webp",
        action="store_true",
        help="输出为 WebP 格式（默认 quality=0.85），节省空间且便于发给 AI 大模型；需安装 Pillow。",
    )
    parser.add_argument(
        "--webp-quality",
        type=float,
        default=0.85,
        metavar="Q",
        help="WebP 压缩质量 0.0～1.0，默认 0.85；仅当 --webp 时生效。",
    )

    args = parser.parse_args()

    ffdec_path = Path(args.ffdec_path).expanduser().resolve()
    if not ffdec_path.exists():
        print(f"[错误] 指定的 FFDec 路径不存在：{ffdec_path}")
        sys.exit(1)

    if args.resources_dir:
        resources_dir = Path(args.resources_dir).expanduser().resolve()
    else:
        resources_dir = _get_resources_dir()

    if not resources_dir.exists():
        print(f"[错误] resources 目录不存在：{resources_dir}")
        sys.exit(1)

    portraits_dir = resources_dir / "flashswf" / "portraits"
    if not portraits_dir.exists():
        print(f"[错误] 未找到 portraits 目录：{portraits_dir}")
        sys.exit(1)

    illustration_dir = portraits_dir / "illustration"
    temp_root = portraits_dir / "_tmp_export_frames"
    temp_root.mkdir(parents=True, exist_ok=True)

    print(f"[信息] 使用 FFDec 路径：{ffdec_path}")
    print(f"[信息] resources 目录：{resources_dir}")
    print(f"[信息] portraits 目录：{portraits_dir}")
    print(f"[信息] illustration 输出目录：{illustration_dir}")
    print(f"[信息] 临时帧导出目录：{temp_root}")

    # 收集所有 portraits 下的 SWF
    swf_files = sorted(portraits_dir.glob("*.swf"))
    if args.only_npc:
        target = args.only_npc.strip()
        swf_files = [p for p in swf_files if p.stem == target]
        if not swf_files:
            print(
                f"[错误] 未在 {portraits_dir} 下找到 NPC '{target}' 对应的 SWF 文件（{target}.swf）。"
            )
            sys.exit(1)

    crop_rect = None
    if args.crop:
        parts = [p.strip() for p in args.crop.split(",")]
        if len(parts) != 4:
            print("[错误] --crop 需要 4 个整数，格式为 left,top,right,bottom，例如 20,30,900,400")
            sys.exit(1)
        try:
            crop_rect = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
        except ValueError:
            print("[错误] --crop 的四个值必须为整数")
            sys.exit(1)

    if (crop_rect is not None or args.smooth) and Image is None:
        print("[错误] 使用 --crop 或 --smooth 需要安装 Pillow：pip install Pillow")
        sys.exit(1)

    if args.webp and Image is None:
        print("[错误] 使用 --webp 需要安装 Pillow：pip install Pillow")
        sys.exit(1)

    result = run_extract(
        ffdec_path=ffdec_path,
        resources_dir=resources_dir,
        overwrite=args.overwrite,
        zoom=args.zoom,
        smooth=args.smooth,
        crop_rect=crop_rect,
        only_npc=args.only_npc,
        webp=args.webp,
        webp_quality=args.webp_quality,
    )

    if not result["success"]:
        print(f"[错误] {result.get('error', '未知错误')}")
        sys.exit(1)
    print(f"\n===== 全部处理完成，成功处理 SWF 数量：{result['processed']} / {result['total']} =====")


if __name__ == "__main__":
    main()

