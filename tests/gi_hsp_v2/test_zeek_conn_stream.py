from pathlib import Path

import pytest

from preprocessing.gi_hsp_v2.zeek_conn_stream import (
    iter_zeek_conn,
    locate_conn_log,
)


def write_log(path, body):
    Path(path).write_text(body, encoding="utf-8")


def test_rows_are_streamed(tmp_path):
    path = tmp_path / "conn.log"
    write_log(
        path,
        "\n".join([
            "#separator \\x09",
            "#fields\tts\tuid\tproto",
            "#types\ttime\tstring\tenum",
            "1.25\tC1\ttcp",
            "2.50\tC2\tudp",
            "#close 3.0",
            "",
        ]),
    )

    iterator = iter_zeek_conn(path)

    assert not isinstance(iterator, list)
    assert list(iterator) == [
        {"ts": "1.25", "uid": "C1", "proto": "tcp"},
        {"ts": "2.50", "uid": "C2", "proto": "udp"},
    ]


def test_row_before_header_is_rejected(tmp_path):
    path = tmp_path / "conn.log"
    write_log(path, "1.25\tC1\ttcp\n")

    with pytest.raises(
        ValueError,
        match="before #fields",
    ):
        list(iter_zeek_conn(path))


def test_missing_header_is_rejected(tmp_path):
    path = tmp_path / "conn.log"
    write_log(path, "#separator \\x09\n#close 3.0\n")

    with pytest.raises(
        ValueError,
        match="header not found",
    ):
        list(iter_zeek_conn(path))


def test_column_mismatch_is_rejected(tmp_path):
    path = tmp_path / "conn.log"
    write_log(
        path,
        "#fields\tts\tuid\tproto\n1.25\tC1\n",
    )

    with pytest.raises(
        ValueError,
        match="column mismatch",
    ):
        list(iter_zeek_conn(path))


def test_duplicate_fields_are_rejected(tmp_path):
    path = tmp_path / "conn.log"
    write_log(
        path,
        "#fields\tts\tuid\tuid\n1.25\tC1\tC2\n",
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        list(iter_zeek_conn(path))


def test_locate_conn_log(tmp_path):
    directory = tmp_path / "nested"
    directory.mkdir()
    path = directory / "conn.log"
    path.write_text("", encoding="utf-8")

    assert locate_conn_log(tmp_path) == path


@pytest.mark.parametrize("count", [0, 2])
def test_locate_requires_exactly_one_log(
    tmp_path,
    count,
):
    for index in range(count):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "conn.log").write_text(
            "",
            encoding="utf-8",
        )

    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        locate_conn_log(tmp_path)
