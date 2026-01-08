# Performance Optimization Strategy

## Edge Deployment on NVIDIA Jetson

---

# 1. PERFORMANCE TARGETS

## Latency Targets (Jetson Orin NX)

| Metric | Target | Maximum | Notes |
|--------|--------|---------|-------|
| End-to-end per package | 500ms | 2000ms | Trigger to decision |
| Per-camera inference | 25ms | 40ms | TensorRT FP16 |
| Capture (5 cameras) | 100ms | 150ms | Synchronized |
| Fusion + decision | 20ms | 50ms | CPU-bound |
| Evidence write | 100ms | 200ms | NVMe SSD |

## Throughput Targets

| Metric | Target | Peak | Notes |
|--------|--------|------|-------|
| Packages/minute | 12 | 20 | Sustained operation |
| Inspections/hour | 720 | 1200 | 8-hour shift |
| Camera frames/second | 5 (per camera) | 10 | During capture burst |

## Degradation Thresholds

| Load Level | Behavior |
|------------|----------|
| Normal (<70% GPU) | Full pipeline, all features |
| Elevated (70-85% GPU) | Disable composite image |
| High (85-95% GPU) | Reduce to 3 priority cameras |
| Critical (>95% GPU) | Queue inspections, alert |

---

# 2. MODEL FORMAT & INFERENCE

## Preferred Format

| Stage | Format | Reason |
|-------|--------|--------|
| Training | PyTorch (.pt) | Standard YOLOv5 |
| Export | ONNX (.onnx) | Intermediate |
| Deployment | TensorRT (.engine) | Optimized for Jetson |

## Precision Strategy

| Precision | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| FP32 | 1× | Baseline | Training only |
| FP16 | 2× | ~0.1% loss | **Production default** |
| INT8 | 3× | ~1% loss | High-throughput mode |

**Recommendation**: FP16 for balance of speed and accuracy.

## Inference Strategy

| Approach | Latency | Throughput | Choice |
|----------|---------|------------|--------|
| Sequential (1 image) | Lowest | Lower | **Selected** |
| Batch (5 images) | Higher | Higher | Not suitable |

**Rationale**: Sequential minimizes per-package latency. Batching increases throughput but delays first result.

## Task Allocation

| Task | Processor | Reason |
|------|-----------|--------|
| Inference | GPU | Maximizes performance |
| Preprocessing | GPU | CUDA resize/normalize |
| NMS | CPU | Small workload |
| Fusion/decision | CPU | Logic, not compute |
| Evidence storage | CPU + DMA | I/O bound |

---

# 3. CAMERA & PIPELINE

## Capture Strategy

| Option | Approach | Trade-off |
|--------|----------|-----------|
| Single frame | 1 capture on trigger | Lowest latency |
| Burst (3 frames) | Pick sharpest | +50ms, better quality |

**Recommendation**: Single frame for speed; burst only if motion blur detected.

## Resolution Trade-offs

| Resolution | Inference Time | Detection Quality |
|------------|----------------|-------------------|
| 1920×1080 (native) | 45ms | Best |
| 1280×720 (resized) | 28ms | Good |
| 640×640 (model input) | 25ms | **Selected** (direct resize) |

**Optimization**: Capture at 1280×720, letterbox to 640×640.

## Preprocessing Pipeline

```
Camera → JPEG decode (CPU) → CUDA upload → 
GPU resize → Normalize → Inference → 
NMS (CPU) → Results
```

| Step | Time | Optimization |
|------|------|--------------|
| JPEG decode | 5ms | Hardware NVJPEG |
| CUDA upload | 2ms | Pinned memory |
| Resize | 1ms | CUDA kernel |
| Normalize | 0.5ms | Fused with resize |
| Total preprocess | ~8ms | Per camera |

## Memory Management

| Resource | Allocation | Strategy |
|----------|------------|----------|
| GPU memory | 2GB reserved | Static allocation |
| Frame buffers | 3 per camera | Ring buffer |
| Inference tensors | Pre-allocated | Reused per inference |
| Evidence images | Streaming | Direct to NVMe |

---

# 4. SCHEDULING & RESOURCES

## Inference Scheduling

```
Trigger → Capture all cameras (parallel)
        → Queue: [CAM-01, CAM-02, CAM-03, CAM-04, CAM-05]
        → Inference: Sequential on GPU
        → Results: Collect all → Fusion
```

| Priority | Camera | Reason |
|----------|--------|--------|
| 1 | CAM-01 (Top) | Best overview |
| 2 | CAM-02 (Front) | Shipping label visible |
| 3 | CAM-03-05 | Additional views |

## GPU Memory Protection

| Mechanism | Implementation |
|-----------|----------------|
| Static allocation | Allocate at startup, never release |
| Memory limit | Cap at 80% of available |
| OOM prevention | Reject new inference if over limit |
| Monitoring | Track CUDA memory every 100ms |

## CPU/IO Isolation

| Thread Pool | Tasks | Cores |
|-------------|-------|-------|
| Capture | Camera I/O | 2 cores |
| Inference | GPU dispatch, NMS | 2 cores |
| Logic | Fusion, decision | 1 core |
| Storage | Evidence write | 1 core |

---

# 5. GRACEFUL DEGRADATION

## Latency Exceeded

| Threshold | Action |
|-----------|--------|
| Inference >40ms | Log warning |
| Inference >60ms | Skip low-priority cameras |
| Total >2000ms | Force decision with available |
| Total >5000ms | Abort, flag SYSTEM_ERROR |

## GPU Saturation

| Utilization | Action |
|-------------|--------|
| 70-85% | Disable composite generation |
| 85-95% | Process 3 cameras only |
| >95% | Queue inspections (max 3) |
| Queue full | REJECT all, alert supervisor |

## Frame Drop Handling

| Condition | Action |
|-----------|--------|
| 1 camera drops | Use remaining 4 |
| 2 cameras drop | Proceed, flag degraded |
| 3+ cameras drop | Force REVIEW_REQUIRED |
| All cameras fail | ABORT inspection |

## Safety Priority

```
ALWAYS: Safety > Throughput

IF uncertain due to degradation:
    → REVIEW_REQUIRED (never auto-ACCEPT)
    
IF system overloaded:
    → Queue or REJECT (never skip inspection)
```

---

# 6. MONITORING & TELEMETRY

## Local Metrics

| Metric | Frequency | Storage |
|--------|-----------|---------|
| Inference latency (per camera) | Every inspection | Rolling 1000 |
| GPU utilization | 100ms | Rolling 5 min |
| GPU memory used | 100ms | Rolling 5 min |
| CPU utilization | 1s | Rolling 5 min |
| Queue depth | Every inspection | Current |
| Frame drop count | Every inspection | Daily total |

## Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Avg inference latency | >35ms | >50ms |
| P95 inference latency | >50ms | >80ms |
| GPU utilization | >80% | >95% |
| GPU memory | >75% | >90% |
| Frame drop rate | >1% | >5% |
| Queue depth | >2 | >5 |

## Regression Detection

| Method | Trigger |
|--------|---------|
| Moving average | Latency up 20% over 1 hour |
| Trend analysis | 3 consecutive hours of increase |
| Anomaly detection | Sudden spike >2× baseline |

---

# 7. BENCHMARKING PLAN

## Hardware Verification

| Test | Method | Pass Criteria |
|------|--------|---------------|
| Cold start | Time from power to ready | <60s |
| Model load | Load TensorRT engine | <5s |
| Single inference | 100 images, median | <30ms |
| Sustained load | 1000 inspections | <35ms avg |

## Real-Time Readiness

| Test | Duration | Pass Criteria |
|------|----------|---------------|
| 12 pkg/min sustained | 1 hour | No queue overflow |
| Peak load (20 pkg/min) | 10 min | <5% degradation |
| Camera failure | Inject failure | Graceful handling |
| GPU stress | Parallel load | No OOM |

## Benchmarking Script

```
1. Warm up: 50 inspections
2. Measure: 500 inspections
3. Record: Per-camera latency, total latency, GPU util
4. Calculate: Mean, P50, P95, P99
5. Compare: Against targets
6. Report: Pass/Fail with details
```

## Optimization Verification

| Optimization | Before | After | Gain |
|--------------|--------|-------|------|
| TensorRT vs PyTorch | 80ms | 25ms | 3.2× |
| FP16 vs FP32 | 45ms | 25ms | 1.8× |
| Pinned memory | 35ms | 28ms | 1.25× |
| Fused preprocess | 32ms | 28ms | 1.14× |

---

## Summary: Optimized Pipeline

```
         ┌─────────────────────────────────────┐
         │           TRIGGER (0ms)             │
         └─────────────────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────┐
         │      PARALLEL CAPTURE (100ms)       │
         │   CAM-01  CAM-02  CAM-03  CAM-04   │
         └─────────────────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────┐
         │    SEQUENTIAL INFERENCE (125ms)     │
         │   25ms × 5 cameras (TensorRT FP16)  │
         └─────────────────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────┐
         │      FUSION + DECISION (20ms)       │
         └─────────────────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────┐
         │      EVIDENCE STORAGE (100ms)       │
         │         (async, non-blocking)       │
         └─────────────────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────┐
         │      TOTAL: ~345ms (target 500ms)   │
         └─────────────────────────────────────┘
```
