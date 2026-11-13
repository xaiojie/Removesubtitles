
import json, io
from pathlib import Path
import pytest
import app as web
from task_options import prepare_task

SUB=[{"type":"subtitle","roi":[10,10,100,80]}]
WATER=[{"type":"watermark","roi":[150,20,250,100]}]

@pytest.mark.parametrize("regions,expected",[(SUB,["clear"]),(WATER,["propainter"]),(SUB+WATER,["clear","propainter"])])
def test_routing_depends_on_regions(regions,expected):
    task=prepare_task(regions,"{}","auto")
    assert task["pipeline"]==expected
    assert task["engine"]==("hybrid" if len(expected)>1 else expected[0])

def test_nested_options_share_time_and_export():
    raw={"start_seconds":2,"duration_seconds":3,"export_crf":14,"clear":{"steps":7},"propainter":{"mask_dilation":9}}
    task=prepare_task(SUB+WATER,json.dumps(raw),"auto")
    assert task["inference"]["clear"]["steps"]==7
    assert task["inference"]["propainter"]["mask_dilation"]==9
    for engine in task["pipeline"]:
        assert task["inference"][engine]["start_seconds"]==2
        assert task["inference"][engine]["duration_seconds"]==3
        assert task["inference"][engine]["export_crf"]==14

def test_manual_legacy_engine_cannot_override_region_routing():
    task=prepare_task(SUB,'{"mask_dilation":12,"duration_seconds":1}',"propainter")
    assert task["pipeline"]==["clear"]
    assert task["inference"]["clear"]["duration_seconds"]==1
    task=prepare_task(WATER,'{"steps":9,"export_crf":14}',"clear")
    assert task["pipeline"]==["propainter"]
    assert task["inference"]["propainter"]["export_crf"]==14

@pytest.mark.parametrize("raw",['{"unknown":1}','{"clear":{"mask_dilation":8}}','{"start_seconds":true}','{"export_crf":99}'])
def test_invalid_nested_options(raw):
    with pytest.raises(ValueError):prepare_task(SUB,raw,"auto")

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

def post(client,regions,engine="auto",inference="{}"):
    return client.post("/api/jobs",data={
        "video":(io.BytesIO(b"video"),"sample.mp4"),"engine":engine,
        "dimensions":"[300,200]","regions":json.dumps(regions),"inference":inference,
    },content_type="multipart/form-data")

def test_api_persists_automatic_mixed_pipeline(client):
    browser,folder=client
    response=post(browser,SUB+WATER,"clear")
    assert response.status_code==202
    config=json.loads((folder/response.json["id"]/"config.json").read_text())
    assert config["pipeline"]==["clear","propainter"] and config["engine"]=="hybrid"

def test_only_required_model_readiness_matters(client,monkeypatch):
    browser,_=client
    monkeypatch.setattr(web,"propainter_status",lambda:(False,"missing"))
    assert post(browser,SUB).status_code==202
    web.jobs.clear()
    assert post(browser,WATER).status_code==503

def test_retry_uses_saved_task_and_preserves_config(client):
    browser,folder=client
    response=post(browser,SUB+WATER)
    job_id=response.json["id"]
    web.jobs[job_id]["state"]="failed"
    result=browser.post(f"/api/jobs/{job_id}/retry")
    assert result.status_code==202 and result.json["id"]==job_id
    assert web.jobs[job_id]["pipeline"]==["clear","propainter"]
    assert browser.post(f"/api/jobs/{job_id}/retry").status_code==409

def test_failed_saved_task_survives_app_state_reset(client):
    browser,folder=client
    response=post(browser,WATER)
    job_id=response.json["id"]
    web.jobs.clear()
    result=browser.get(f"/api/jobs/{job_id}")
    assert result.status_code==200 and result.json["state"]=="failed"
    assert result.json["can_retry"] is True

def test_invalid_retry_id(client):
    browser,_=client
    assert browser.post("/api/jobs/not-a-job/retry").status_code==404



def test_retry_write_failure_does_not_lock_gate_or_truncate_config(client,monkeypatch):
    browser,folder=client
    response=post(browser,SUB+WATER)
    job_id=response.json["id"]
    web.jobs[job_id]["state"]="failed"
    config=folder/job_id/"config.json"
    original=config.read_text()
    def fail_write(*args,**kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(web,"save_configuration",fail_write,raising=False)
    response=browser.post(f"/api/jobs/{job_id}/retry")
    assert response.status_code==500
    assert web.jobs[job_id]["state"]=="failed"
    assert web.jobs[job_id]["can_retry"] is True
    assert config.read_text()==original



def test_resume_second_stage_keeps_success_signature_and_needs_only_pending_model(client,monkeypatch):
    from pipeline_worker import signature, completed_stages
    browser,root=client
    response=post(browser,SUB+WATER)
    job_id=response.json["id"]
    folder=root/job_id
    config=json.loads((folder/"config.json").read_text())
    stage=folder/"stages/clear";stage.mkdir(parents=True)
    (stage/"output.mkv").write_bytes(b"lossless output")
    (stage/"success.json").write_text(json.dumps({"stage":"clear","artifact":"output.mkv","config_sha256":signature(config)}))
    web.jobs[job_id]["state"]="failed"
    monkeypatch.setattr(web,"model_status",lambda:(False,"missing"))
    assert browser.post(f"/api/jobs/{job_id}/retry").status_code==202
    assert completed_stages(folder)==["clear"]

def test_retry_thread_failure_releases_running_gate(client,monkeypatch):
    browser,root=client
    response=post(browser,WATER)
    job_id=response.json["id"]
    web.jobs[job_id]["state"]="failed"
    class BrokenThread:
        def __init__(self,**kwargs):pass
        def start(self):raise RuntimeError("thread unavailable")
    monkeypatch.setattr(web,"Thread",BrokenThread)
    assert browser.post(f"/api/jobs/{job_id}/retry").status_code==500
    assert web.jobs[job_id]["state"]=="failed"
    assert web.jobs[job_id]["can_retry"] is True

