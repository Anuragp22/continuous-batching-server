def test_healthz_returns_ok(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_serves_prometheus_text(client) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert b"cbserver_requests_total" in response.content


def test_unknown_route_returns_404(client) -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
