import io
import json
from dataclasses import asdict
import pytest
import app as web
from propainter_options import ProPainterOptions

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(web,"DATA",tmp_path)
    monkeypatch.setattr(web,"model_status",lambda:(True,"ready"))
    monkeypatch.setattr(web,"propainter_status",lambda:(True,"ready"))
    class FakeThread:
        def __init__(self,**kwargs):pass
        def start(self):pass
    monkeypatch.setattr(web,"Thread",FakeThread)
    web.jobs.clear()
    return web.app.test_client(),tmp_path

def form(engine="propainter",inference="{}"):
    regions = [{"type":"subtitle","roi":[100,600,1000,700]}] if engine=="clear" else [{"type":"watermark","roi":[10,10,250,150]}]
    return {"video":(io.BytesIO(b"video"),"sample.mp4"),"engine":engine,"dimensions":"[1280,720]","regions":json.dumps(regions),"inference":inference}

def test_propainter_submission_saves_engine_and_typed_defaults(client):
    browser,folder=client
    response=browser.post("/api/jobs",data=form(),content_type="multipart/form-data")
    assert response.status_code==202
    config=json.loads((folder/response.json["id"]/"config.json").read_text())
    assert config["engine"]=="propainter"
    assert config["inference"]["propainter"]["mask_dilation"]==8
    assert config["inference"]["propainter"]["fp16"] is True
    assert "steps" not in config["inference"]["propainter"]

def test_legacy_engine_less_request_remains_clear(client):
    browser,folder=client
    data=form("clear");data.pop("engine")
    response=browser.post("/api/jobs",data=data,content_type="multipart/form-data")
    assert response.status_code==202
    config=json.loads((folder/response.json["id"]/"config.json").read_text())
    assert config["engine"]=="clear" and config["inference"]["clear"]["steps"]==5

def test_cross_engine_job_gate(client):
    browser,_=client
    assert browser.post("/api/jobs",data=form("clear"),content_type="multipart/form-data").status_code==202
    assert browser.post("/api/jobs",data=form(),content_type="multipart/form-data").status_code==409

@pytest.mark.parametrize("engine,inference",[("unknown","{}"),("propainter","{\"steps\":5}"),("propainter","{\"fp16\":\"false\"}"),("propainter","{\"mask_dilation\":true}")])
def test_invalid_engine_or_options_rejected(client,engine,inference):
    browser,_=client
    assert browser.post("/api/jobs",data=form(engine,inference),content_type="multipart/form-data").status_code==400

def test_propainter_ready_is_independent_of_clear(client,monkeypatch):
    browser,_=client
    monkeypatch.setattr(web,"model_status",lambda:(False,"missing"))
    assert browser.post("/api/jobs",data=form(),content_type="multipart/form-data").status_code==202

def test_defaults_and_worker_dispatch(client,monkeypatch):
    browser,_=client
    monkeypatch.setattr(web,"gpu_status",lambda:{"name":"L20"})
    system=browser.get("/api/system").json
    assert system["default_engine"]=="auto"
    assert system["propainter_defaults"]["fp16"] is True
    command=web.worker_command(web.DATA/"job")
    assert str(command[2]).endswith("pipeline_worker.py")
    assert str(command[-1]).endswith("propainter-eval/venv/bin/python")
    assert system["auto_routing"] is True

def test_options_keep_time_window_and_strict_types():
    options=ProPainterOptions.parse("{\"start_seconds\":1,\"duration_seconds\":2}")
    assert options.frame_range(30,300)==(30,90)
    assert asdict(options)["watermark_mask_mode"]=="auto"
    for bad in [{"mask_dilation":-1},{"neighbor_length":1},{"fp16":1},{"duration_seconds":0},{"resolution":"small"},{"raft_iter":0},{"watermark_mask_mode":"sample"}]:
        with pytest.raises(ValueError):ProPainterOptions.parse(json.dumps(bad))
