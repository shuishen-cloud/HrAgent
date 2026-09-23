"""简历解析：上传的文件 → 纯文本，供 LLM 据此出题。

支持 .pdf / .docx / .txt / .md。解析库都在函数内懒加载，
所以单测不装 pypdf / python-docx 也能跑（打桩 _read_* 即可）。
"""
from __future__ import annotations

import io
import re
from pathlib import Path

from . import config

SUPPORTED_EXTS = (".pdf", ".docx", ".txt", ".md")

# 简历常见的表格排版，docx 里表格和正文是分开存的，两边都要取
_DOCX_TABLE_SEP = " | "


class ResumeError(Exception):
    """解析失败，消息可直接展示给用户。"""


def _read_pdf(data: bytes) -> str:
    """用 pypdfium2（PDFium，Chrome 的 PDF 引擎）解析。

    **不要换回 pypdf。** 实测：当上传的文件其实不是 PDF 时（比如扩展名改了、
    下载损坏），pypdf 会**间歇性段错误**把整个进程打掉（连跑 20 次崩 2 次；
    非 PDF 的样本 8/8 崩），而这是用户上传路径 —— 崩的是服务，不是这一次请求。
    pypdfium2 遇到同样的输入只会干净地抛 PdfiumError，可以捕获成可读提示。
    """
    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(data)
    except Exception as exc:
        raise ResumeError(f"PDF 打不开：{exc}") from exc

    try:
        pages = [page.get_textpage().get_text_range() for page in doc]
    except Exception as exc:
        raise ResumeError(f"PDF 内容读取失败：{exc.__class__.__name__}") from exc
    return "\n".join(pages)


def _read_docx(data: bytes) -> str:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ResumeError(f"Word 解析失败：{exc.__class__.__name__}") from exc

    parts = [p.text for p in document.paragraphs]
    for table in document.tables:          # 简历常用表格排版，漏了就只剩半份
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(_DOCX_TABLE_SEP.join(cells))
    return "\n".join(parts)


def _read_text(data: bytes) -> str:
    """纯文本。中文简历常是 GBK/GB18030，UTF-8 解不出来时再试。"""
    for encoding in ("utf-8", "gb18030", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ResumeError("文本编码识别不了（不是 UTF-8 / GBK / UTF-16）")


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_text,
    ".md": _read_text,
}


def _clean(text: str) -> str:
    """压掉多余空白：PDF 抽出来的文字常有大片空行和行尾空格。"""
    lines = [re.sub(r"[ \t　]+", " ", ln).strip() for ln in text.splitlines()]
    kept: list[str] = []
    for ln in lines:
        if ln or (kept and kept[-1]):     # 连续空行只留一个
            kept.append(ln)
    return "\n".join(kept).strip()


def extract_text(filename: str, data: bytes) -> str:
    """把上传的简历文件解析成纯文本。

    失败一律抛 ResumeError（消息可直接给用户看），不返回半截内容 ——
    拿半份简历去出题，比明确报错更糟。
    """
    if not data:
        raise ResumeError("文件是空的")

    if len(data) > config.RESUME_MAX_BYTES:
        raise ResumeError(
            f"文件太大了（{len(data) // 1024 // 1024} MB），"
            f"上限 {config.RESUME_MAX_BYTES // 1024 // 1024} MB"
        )

    suffix = Path(filename or "").suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        raise ResumeError(
            f"不支持 {suffix or '这种'} 格式，请上传 "
            + " / ".join(SUPPORTED_EXTS)
        )

    text = _clean(reader(data))

    if len(text) < config.RESUME_MIN_CHARS:
        # 最常见的原因：扫描件 / 图片版 PDF，里面根本没有文字层
        hint = "（可能是扫描件或图片版，里面没有可提取的文字，需要先做 OCR）" \
            if suffix == ".pdf" else ""
        raise ResumeError(f"没解析出有效文字{hint}")

    # 超长简历截断，避免把上下文撑爆
    if len(text) > config.RESUME_MAX_CHARS:
        text = text[: config.RESUME_MAX_CHARS] + "\n…（简历过长，已截断）"
    return text


def summarize(text: str, limit: int = 60) -> str:
    """给界面显示的一行预览。"""
    one_line = re.sub(r"\s+", " ", text).strip()
    return one_line[:limit] + ("…" if len(one_line) > limit else "")
