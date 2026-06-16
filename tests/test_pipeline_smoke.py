from src.pipeline import run_pipeline


def test_pipeline_smoke_frame():
    result = run_pipeline("013342")

    assert result["world"]["timestamp"] == "013342"
    assert result["decision"]["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert result["plan"]["action"]
    assert len(result["scene"]["navigation"]["desired_path"]) >= 2
    assert result["plan"]["navigation"]["desired_path"]
