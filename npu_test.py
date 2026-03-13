#!/usr/bin/env python3
import sys
import time
import numpy as np
import openvino as ov
from openvino.runtime import opset8 as ops


def build_static_model():
    x = ops.parameter([1, 4], np.float32, name="x")
    c = ops.constant(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    y = ops.add(x, c)
    y = ops.relu(y)
    return ov.Model([y], [x], "npu_smoke")


def main():
    core = ov.Core()
    devices = core.available_devices
    print("Available devices:", devices)

    npu_devices = [d for d in devices if d.split(".")[0] == "NPU"]
    if not npu_devices:
        print("FAIL: No NPU device is visible to OpenVINO.")
        sys.exit(1)

    print("NPU visible:", npu_devices)

    model = build_static_model()

    t0 = time.perf_counter()
    compiled_model = core.compile_model(model, "NPU")
    t1 = time.perf_counter()

    print(f"Compiled on NPU in {t1 - t0:.3f}s")

    infer_request = compiled_model.create_infer_request()
    x = np.array([[1.0, -2.0, 0.5, 10.0]], dtype=np.float32)
    result = infer_request.infer({"x": x})

    output_port = compiled_model.output(0)
    y = result[output_port]

    print("Input :", x)
    print("Output:", y)
    print("PASS: NPU compile + inference worked.")


if __name__ == "__main__":
    main()
