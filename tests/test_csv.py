"""
Tests for robust CSV ingestion (encoding fallback).
Regression test: sales_data_sample.csv is cp1252-encoded and must upload cleanly.
"""

import os

import pytest

from src.utils.csv import read_csv_bytes


def test_utf8_csv_parses():
    df = read_csv_bytes(b"a,b\n1,2\n3,4\n", "t.csv")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_utf8_bom_csv_parses():
    df = read_csv_bytes("a,b\n1,2\n".encode("utf-8-sig"), "t.csv")
    assert list(df.columns) == ["a", "b"]


def test_cp1252_csv_falls_back():
    # Byte 0x84 is invalid UTF-8 but valid cp1252 (U+201E). Mirrors the reported
    # "'utf-8' codec can't decode byte 0x84" failure.
    raw = b"city,country\nBerguvsv\x84gen,Sweden\n"
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    df = read_csv_bytes(raw, "t.csv")
    assert len(df) == 1
    assert df["country"].iloc[0] == "Sweden"


def test_latin1_csv_parses():
    raw = "name\ncaf\xe9\n".encode("latin-1")
    df = read_csv_bytes(raw, "t.csv")
    assert len(df) == 1


def test_empty_file_rejected():
    with pytest.raises(ValueError, match="empty"):
        read_csv_bytes(b"   ", "t.csv")


def test_repo_sample_file_parses():
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    sample = os.path.join(repo_root, "sales_data_sample.csv")
    with open(sample, "rb") as fh:
        df = read_csv_bytes(fh.read(), "sales_data_sample.csv")
    assert len(df) > 1000
    assert len(df.columns) > 5
