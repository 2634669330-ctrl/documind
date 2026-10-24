"""解析器的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.parser import ParseError, UnsupportedFormatError, parse_file


def test_parse_txt(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("第一行内容。\n第二行内容。", encoding="utf-8")

    doc = parse_file(path)
    assert doc.filename == "note.txt"
    assert len(doc.pages) == 1
    assert "第一行内容" in doc.pages[0].text


def test_parse_markdown_with_front_heading(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# 标题\n\n正文一段。\n\n正文二段。", encoding="utf-8")

    doc = parse_file(path)
    assert "正文一段" in doc.text


def test_parse_gb18030_fallback(tmp_path: Path) -> None:
    path = tmp_path / "gbk.txt"
    path.write_bytes("中文编码测试。".encode("gb18030"))

    doc = parse_file(path)
    assert "中文编码测试" in doc.text


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError):
        parse_file(path)


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  ", encoding="utf-8")

    with pytest.raises(ParseError):
        parse_file(path)


def test_long_txt_split_into_pseudo_pages(tmp_path: Path) -> None:
    path = tmp_path / "long.txt"
    path.write_text("内容。" * 2000, encoding="utf-8")  # 6000 字 -> 至少 2 个逻辑页

    doc = parse_file(path)
    assert len(doc.pages) >= 2
    assert [p.no for p in doc.pages] == list(range(1, len(doc.pages) + 1))
