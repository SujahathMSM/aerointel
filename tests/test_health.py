async def test_health_live(client):
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_request_id_returned(client):
    response = await client.get("/health/live")
    assert response.headers["x-request-id"]


async def test_query_validates_input(client):
    # empty question must be rejected by the schema, never reach services
    response = await client.post("/v1/query", json={"question": ""})
    assert response.status_code == 422
