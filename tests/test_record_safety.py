import pytest
from warehouse.provenance import guard_output_directory


def test_recorded_run_is_not_overwritten(tmp_path):
    record = tmp_path / "result.json"
    record.write_text('{"success": true}')
    with pytest.raises(ValueError, match="not empty"):
        guard_output_directory(tmp_path)
    assert record.read_text() == '{"success": true}'


def test_resume_requires_run_identity(tmp_path):
    with pytest.raises(ValueError, match="existing run"):
        guard_output_directory(tmp_path, resume=True)
    (tmp_path / "config.json").write_text('{}')
    (tmp_path / "metadata.json").write_text('{}')
    guard_output_directory(tmp_path, resume=True)


def test_new_directory_is_allowed(tmp_path):
    guard_output_directory(tmp_path / "fresh")
    guard_output_directory(tmp_path)
