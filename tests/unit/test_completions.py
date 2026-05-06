import json


def test_non_stream_returns_completion_shape(client) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "qwen", "prompt": "hello", "max_tokens": 4, "stream": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "qwen"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["text"] == "tok0 tok1 tok2 tok3 "
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 4
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 4


def test_invalid_request_returns_422(client) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "qwen", "prompt": "", "max_tokens": 4},
    )
    assert response.status_code == 422


def test_stream_emits_sse_chunks_and_done(client) -> None:
    with client.stream(
        "POST",
        "/v1/completions",
        json={"model": "qwen", "prompt": "hello", "max_tokens": 3, "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        data_lines = [
            line[len("data: "):]
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert data_lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert len(chunks) == 4
    assert chunks[0]["object"] == "text_completion"
    assert chunks[0]["choices"][0]["text"] == "tok0 "
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_non_stream_honors_stop_sequence(client) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "qwen",
            "prompt": "hi",
            "max_tokens": 10,
            "stop": ["tok2"],
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"] == "tok0 tok1 "
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 2


def test_stream_honors_stop_sequence(client) -> None:
    import json

    with client.stream(
        "POST",
        "/v1/completions",
        json={
            "model": "qwen",
            "prompt": "hi",
            "max_tokens": 10,
            "stop": ["tok1"],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        data_lines = [
            line[len("data: "):]
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    chunks = [json.loads(line) for line in data_lines if line != "[DONE]"]
    text_chunks = [c["choices"][0]["text"] for c in chunks if c["choices"][0]["text"]]
    assert text_chunks == ["tok0 "]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_metrics_count_completed_request(client) -> None:
    pre = client.get("/metrics").content
    client.post(
        "/v1/completions",
        json={"model": "qwen", "prompt": "hi", "max_tokens": 2, "stream": False},
    )
    post = client.get("/metrics").content
    assert b'cbserver_requests_total{status="completed"}' in post
    assert post.count(b"cbserver_decode_tokens_total") >= pre.count(b"cbserver_decode_tokens_total")
