import asyncio
import http
import logging
import numpy as np
import time
import traceback

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from ..utils.standardlizations import get_standardization

input_transform = get_standardization('piper')['input']

logger = logging.getLogger(__name__)


class WebsocketDummyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        host = "0.0.0.0",
        port = None,
        mode = '+x',
        delta = 1000,
        use_multi_arm = True,
    ) -> None:
        self._host = host
        self._port = port
        self.mode = mode
        self.use_multi_arm = use_multi_arm

        if mode == '+x':
            self.action = [delta, 0, 0, 0, 0, 0, 0]
        elif mode == '-x':
            self.action = [-delta, 0, 0, 0, 0, 0, 0]
        elif mode == '+y':
            self.action = [0, delta, 0, 0, 0, 0, 0]
        elif mode == '-y':
            self.action = [0, -delta, 0, 0, 0, 0, 0]
        elif mode == '+z':
            self.action = [0, 0, delta, 0, 0, 0, 0]
        elif mode == '-z':
            self.action = [0, 0, -delta, 0, 0, 0, 0]
        elif mode == '+rx':
            self.action = [0, 0, 0, delta, 0, 0, 0]
        elif mode == '-rx':
            self.action = [0, 0, 0, -delta, 0, 0, 0]
        elif mode == '+ry':
            self.action = [delta, 0, 0, 0, delta, 0, 0]
        elif mode == '-ry':
            self.action = [delta, 0, 0, 0, -delta, 0, 0]
        elif mode == '+rz':
            self.action = [0, 0, 0, 0, 0, delta, 0]
        elif mode == '-rz':
            self.action = [0, 0, 0, 0, 0, -delta, 0]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.action = np.array(self.action)

        if use_multi_arm:
            self.action = np.concatenate([self.action, self.action])

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack({}))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                # action = self._policy.infer(obs)
                action = self.action.copy()
                
                action = input_transform(action)
                action = {
                    "actions": [action],
                }

                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
