"""简历解析测试。

.txt / .docx / .pdf 都造**真实文件**来测，不打桩解析器本身 ——
打桩就测不出「编码解错」「表格漏读」这类真实问题。
"""
from __future__ import annotations

import io

import pytest

from app import config, resume

CN_RESUME = """张伟
求职意向：后端开发工程师
电话：138-0000-0000   邮箱：zhangwei@example.com

教育经历
2019.09 - 2023.06  太原工业学院  软件工程  本科

项目经历
西瓜甜度机器视觉检测   2022.03 - 2022.10
- 负责图像预处理与模型推理
- 使用 OpenCV 做去噪与轮廓提取，轻量 CNN 预测甜度
- 获计算机设计大赛省级奖项

技能
Python / OpenCV / MySQL / Git
"""


def _docx_bytes(paragraphs: list[str], table_rows: list[list[str]] | None = None) -> bytes:
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    if table_rows:
        t = d.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, val in enumerate(row):
                t.cell(r, c).text = val
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- 正常解析

def test_解析_txt_utf8():
    text = resume.extract_text("简历.txt", CN_RESUME.encode("utf-8"))
    assert "张伟" in text
    assert "OpenCV" in text
    assert "太原工业学院" in text


def test_解析_txt_gbk():
    """中文简历常见的 GBK 编码不能变成乱码。"""
    text = resume.extract_text("简历.txt", CN_RESUME.encode("gb18030"))
    assert "张伟" in text
    assert "西瓜甜度" in text


def test_解析_markdown():
    body = "# 王强\n\n求职意向：Python 后端开发\n\n熟悉 FastAPI 与 MySQL，做过订单系统"
    text = resume.extract_text("resume.md", body.encode("utf-8"))
    assert "Python 后端开发" in text


def test_解析_docx_含段落():
    data = _docx_bytes(["李娜", "求职意向：算法工程师", "熟悉 PyTorch 与 OpenCV"])
    text = resume.extract_text("简历.docx", data)
    assert "李娜" in text
    assert "PyTorch" in text


def test_解析_docx_读到表格内容():
    """简历常用表格排版，只读 paragraphs 会漏掉一大半内容。"""
    data = _docx_bytes(
        ["张伟的个人简历", "求职意向：后端开发工程师"],
        [["项目", "西瓜甜度机器视觉检测"], ["职责", "图像预处理与模型推理"]],
    )
    text = resume.extract_text("简历.docx", data)

    assert "西瓜甜度机器视觉检测" in text
    assert "图像预处理与模型推理" in text       # 表格里的内容必须读到


def _minimal_pdf(text: str = "Hello Resume PDF - Python developer with OpenCV") -> bytes:
    """手写一个最小合法 PDF（一页 + 一行文字）。

    不依赖系统里现成的 PDF 样本 —— 那些是否可用因机器而异，测试会变得不稳定。
    """
    content = b"BT /F1 14 Tf 20 100 Td (" + text.encode() + b") Tj ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj\n" + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets[1:]:
        out += ("%010d 00000 n \n" % off).encode()
    out += (b"trailer\n<< /Size " + str(len(objs) + 1).encode() +
            b" /Root 1 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF\n")
    return bytes(out)


def test_解析_pdf():
    text = resume.extract_text("简历.pdf", _minimal_pdf())
    assert "Hello Resume PDF" in text


def test_非_PDF_文件不能把进程搞崩():
    """回归测试，对应一个真实踩过的坑。

    上传一个扩展名是 .pdf、内容却不是 PDF 的文件（改错扩展名、下载损坏都会这样），
    原先用的 pypdf 会**间歇性段错误**把整个进程打掉 —— 实测非 PDF 样本 8/8 崩。
    这是用户上传路径，崩的是服务而不只是这一次请求。

    现在换成 pypdfium2：同样的输入只抛可捕获的异常，转成可读提示。
    这个用例只要「能正常返回 ResumeError」即算通过 —— 进程崩了就根本跑不到断言。
    """
    not_a_pdf = b"% This is a config file for dvips, meant to produce PostScript\n"

    with pytest.raises(resume.ResumeError, match="PDF 打不开"):
        resume.extract_text("伪装.pdf", not_a_pdf)


def test_损坏的_PDF_也只报错不崩():
    """头部像 PDF、后面是垃圾 —— 比完全不是 PDF 更容易触发解析器异常路径。"""
    broken = b"%PDF-1.4\n" + b"\x00\xff\xfe garbage " * 50 + b"\n%%EOF\n"

    with pytest.raises(resume.ResumeError):
        resume.extract_text("损坏.pdf", broken)


# ---------------------------------------------------------------- 错误处理
# 失败一律抛 ResumeError（消息可直接给用户看），绝不返回半截内容 ——
# 拿半份简历去出题比明确报错更糟。

def test_空文件报错():
    with pytest.raises(resume.ResumeError, match="空"):
        resume.extract_text("简历.txt", b"")


def test_不支持的格式说清支持哪些():
    with pytest.raises(resume.ResumeError, match="不支持"):
        resume.extract_text("简历.png", b"\x89PNG\r\n\x1a\n" + b"x" * 100)


def test_没有扩展名也不崩():
    with pytest.raises(resume.ResumeError):
        resume.extract_text("简历", b"whatever")


def test_扫描件_PDF_给出可操作提示():
    """图片版 PDF 里没有文字层，要说清是「需要 OCR」而不是「解析失败」。"""
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)

    with pytest.raises(resume.ResumeError, match="OCR"):
        resume.extract_text("扫描件.pdf", buf.getvalue())


def test_内容太短视为无效():
    with pytest.raises(resume.ResumeError, match="有效文字"):
        resume.extract_text("简历.txt", "你好".encode("utf-8"))


def test_超大文件被拦下(monkeypatch):
    monkeypatch.setattr(config, "RESUME_MAX_BYTES", 100)
    with pytest.raises(resume.ResumeError, match="太大"):
        resume.extract_text("简历.txt", b"x" * 200)


def test_超长简历被截断(monkeypatch):
    monkeypatch.setattr(config, "RESUME_MAX_CHARS", 50)
    text = resume.extract_text("简历.txt", ("很长的内容" * 100).encode("utf-8"))

    assert len(text) < 120
    assert "已截断" in text


# ---------------------------------------------------------------- 清洗与预览

def test_去掉多余空行和行尾空格(monkeypatch):
    # 这个用例只关心清洗，跟长度阈值无关，把阈值放低免得被它绊住
    monkeypatch.setattr(config, "RESUME_MIN_CHARS", 1)

    text = resume.extract_text("a.txt", "第一行   \n\n\n\n第二行\t\n".encode("utf-8"))
    assert text == "第一行\n\n第二行"


def test_预览函数压成一行():
    assert resume.summarize("第一行\n第二行") == "第一行 第二行"
    assert resume.summarize("很长" * 100, limit=10).endswith("…")
