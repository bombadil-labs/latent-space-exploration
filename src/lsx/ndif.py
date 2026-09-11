"""NDIF remote execution through a credential-injecting egress proxy.

Two accommodations for this environment:
  1. The proxy adds the `ndif-api-key` header itself, so the client must NOT send one (it would send
     an empty/None value otherwise). We strip it from every request.
  2. The proxy does not carry WebSocket upgrades, so we use nnsight's non-blocking mode: submit over
     HTTPS, poll `/response/{job_id}` until complete, then push the result into the tracer.
Usage (the trace block must be in a real source file; nnsight captures its source):
    model = LanguageModel("Qwen/Qwen2.5-7B", device_map="auto", dispatch=False)
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(prompt, backend=backend) as tracer:
        h = model.model.layers[14].output[0][0, -1].save()
    backend.wait(tracer)     # polls until the saved values are populated
"""
from __future__ import annotations

import time
import httpx
from nnsight.intervention.backends.remote import RemoteBackend


class ProxyAuthBackend(RemoteBackend):
    """RemoteBackend that omits the API-key header (the proxy injects it) and never opens a WebSocket."""

    def __init__(self, model_key: str, **kw):
        super().__init__(model_key, blocking=False, api_key="proxy", **kw)

    def request(self, tracer):
        data, headers = super().request(tracer)
        headers.pop("ndif-api-key", None)
        return data, headers

    def get_response(self):
        from nnsight.schema.response import ResponseModel
        timeout = httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{self.address}/response/{self.job_id}")
        if response.status_code == 200:
            return self.handle_response(ResponseModel(**response.json()))
        raise Exception(f"{response.status_code} {response.reason_phrase}: {response.text[:200]}")

    def wait(self, tracer, poll: float = 2.0, timeout: float = 420.0, retries: int = 6, resubmits: int = 2):
        """Poll until complete. Transient transport errors are retried with backoff; a job that has not
        completed after `timeout` seconds is resubmitted (up to `resubmits` times) since hung jobs happen."""
        t0 = time.time(); errs = 0
        while True:
            if time.time() - t0 > timeout:
                if resubmits <= 0:
                    raise TimeoutError(f"NDIF job {self.job_id} not complete after {timeout}s")
                resubmits -= 1; self.job_id = None; t0 = time.time()      # resubmit the same tracer
            try:
                result = self(tracer)
                errs = 0
            except (httpx.TransportError, ConnectionError, OSError) as e:
                errs += 1
                if errs > retries:
                    raise
                time.sleep(min(2 ** errs, 60))
                continue
            if result is not None:
                try:
                    tracer.push(result)
                except AttributeError:
                    pass            # trace context already closed; caller reads the raw result
                self.result = result       # dict keyed by the .save()'d variable names
                return result
            time.sleep(poll)
