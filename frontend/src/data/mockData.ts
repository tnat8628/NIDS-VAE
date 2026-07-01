import type { PredictionResponse } from '@/types/api'

/**
 * Dữ liệu giả lập cho phát triển UI trước khi kết nối backend thật.
 * Dựa trên kết quả thực tế từ fixed_batch.csv (128 flows, 13 anomaly).
 */
export const mockPredictionResponse: PredictionResponse = {
  status: 'ok',
  summary: {
    total_flows: 128,
    anomaly_count: 13,
    normal_count: 115,
    anomaly_rate: 0.101562,
    threshold: 3.2182483673095703,
  },
  results: [
    { row_index: 0, reconstruction_error: 0.4729, prediction: 0, prediction_label: 'normal' },
    { row_index: 1, reconstruction_error: 0.2847, prediction: 0, prediction_label: 'normal' },
    { row_index: 2, reconstruction_error: 4.1023, prediction: 1, prediction_label: 'anomaly' },
    { row_index: 3, reconstruction_error: 0.3912, prediction: 0, prediction_label: 'normal' },
    { row_index: 4, reconstruction_error: 5.7841, prediction: 1, prediction_label: 'anomaly' },
  ],
}

export const mockHealthResponse = {
  status: 'ok' as const,
  model_loaded: true,
  scaler_loaded: true,
  threshold_loaded: true,
  service_name: 'NIDS VAE Anomaly Detection',
}

// ---------------------------------------------------------------------------
// Temporary placeholder exports for migrated Lovable chart components.
// Replace with real API-derived data once backend integration is complete.
// ---------------------------------------------------------------------------

export const stats = {
  totalFlows: 128,
  anomalies: 13,
  normal: 115,
  anomalyRate: 10.16,
  threshold: 3.2182,
  modelStatus: 'Online',
  modelVersion: 'vae-nids-v1.4.2',
  latencyMs: 38,
}

export type TimelineBin = {
  hour: string
  normal: number
  anomaly: number
}

export const timeline: TimelineBin[] = [
  { hour: '00:00', normal: 42, anomaly: 0 },
  { hour: '02:00', normal: 38, anomaly: 1 },
  { hour: '04:00', normal: 35, anomaly: 0 },
  { hour: '06:00', normal: 51, anomaly: 2 },
  { hour: '08:00', normal: 89, anomaly: 3 },
  { hour: '10:00', normal: 112, anomaly: 5 },
  { hour: '12:00', normal: 98, anomaly: 4 },
  { hour: '14:00', normal: 105, anomaly: 6 },
  { hour: '16:00', normal: 95, anomaly: 3 },
  { hour: '18:00', normal: 78, anomaly: 2 },
  { hour: '20:00', normal: 60, anomaly: 1 },
  { hour: '22:00', normal: 48, anomaly: 0 },
]

export type HistogramBin = {
  bin: string
  normal: number
  anomaly: number
}

export const histogram: HistogramBin[] = [
  { bin: '0.0', normal: 30, anomaly: 0 },
  { bin: '0.5', normal: 28, anomaly: 0 },
  { bin: '1.0', normal: 22, anomaly: 0 },
  { bin: '1.5', normal: 15, anomaly: 0 },
  { bin: '2.0', normal: 10, anomaly: 0 },
  { bin: '2.5', normal: 6, anomaly: 0 },
  { bin: '3.0', normal: 4, anomaly: 2 },
  { bin: '3.5', normal: 0, anomaly: 4 },
  { bin: '4.0', normal: 0, anomaly: 3 },
  { bin: '4.5', normal: 0, anomaly: 2 },
  { bin: '5.0', normal: 0, anomaly: 2 },
]

export type SeverityBucket = {
  severity: string
  count: number
}

export const severityBuckets: SeverityBucket[] = [
  { severity: 'nghiêm trọng', count: 3 },
  { severity: 'cao', count: 4 },
  { severity: 'trung bình', count: 4 },
  { severity: 'thấp', count: 2 },
]

// ---------------------------------------------------------------------------
// Temporary placeholder exports for migrated Lovable table/insight/health
// components. Replace with real API-derived data once backend integration
// is complete.
// ---------------------------------------------------------------------------

export type FlowMock = {
  id: string
  rowIndex: number
  srcIp: string
  dstIp: string
  port: number
  protocol: string
  bytes: number
  reconstructionError: number
  prediction: 0 | 1
  predictionLabel: 'normal' | 'anomaly'
  severity: 'low' | 'medium' | 'high' | 'critical'
  timestamp: string
}

export const flows: FlowMock[] = [
  { id: 'f001', rowIndex: 0,  srcIp: '192.168.1.10',  dstIp: '10.0.0.5',   port: 80,   protocol: 'TCP',  bytes: 1200,  reconstructionError: 0.4729, prediction: 0, predictionLabel: 'normal',  severity: 'low',      timestamp: '2026-05-28T08:00:00Z' },
  { id: 'f002', rowIndex: 1,  srcIp: '192.168.1.11',  dstIp: '10.0.0.5',   port: 443,  protocol: 'TCP',  bytes: 980,   reconstructionError: 0.2847, prediction: 0, predictionLabel: 'normal',  severity: 'low',      timestamp: '2026-05-28T08:01:00Z' },
  { id: 'f003', rowIndex: 2,  srcIp: '172.16.0.55',   dstIp: '10.0.0.1',   port: 22,   protocol: 'TCP',  bytes: 52000, reconstructionError: 4.1023, prediction: 1, predictionLabel: 'anomaly', severity: 'high',     timestamp: '2026-05-28T08:02:00Z' },
  { id: 'f004', rowIndex: 3,  srcIp: '192.168.1.12',  dstIp: '8.8.8.8',    port: 53,   protocol: 'UDP',  bytes: 120,   reconstructionError: 0.3912, prediction: 0, predictionLabel: 'normal',  severity: 'low',      timestamp: '2026-05-28T08:03:00Z' },
  { id: 'f005', rowIndex: 4,  srcIp: '10.10.0.200',   dstIp: '10.0.0.1',   port: 445,  protocol: 'TCP',  bytes: 98000, reconstructionError: 5.7841, prediction: 1, predictionLabel: 'anomaly', severity: 'critical', timestamp: '2026-05-28T08:04:00Z' },
  { id: 'f006', rowIndex: 5,  srcIp: '192.168.1.13',  dstIp: '10.0.0.5',   port: 80,   protocol: 'TCP',  bytes: 760,   reconstructionError: 0.5103, prediction: 0, predictionLabel: 'normal',  severity: 'low',      timestamp: '2026-05-28T08:05:00Z' },
  { id: 'f007', rowIndex: 6,  srcIp: '172.16.0.88',   dstIp: '10.0.0.2',   port: 3389, protocol: 'TCP',  bytes: 43000, reconstructionError: 3.8812, prediction: 1, predictionLabel: 'anomaly', severity: 'high',     timestamp: '2026-05-28T08:06:00Z' },
  { id: 'f008', rowIndex: 7,  srcIp: '192.168.1.14',  dstIp: '10.0.0.5',   port: 443,  protocol: 'TCP',  bytes: 1500,  reconstructionError: 0.2201, prediction: 0, predictionLabel: 'normal',  severity: 'low',      timestamp: '2026-05-28T08:07:00Z' },
  { id: 'f009', rowIndex: 8,  srcIp: '10.10.0.201',   dstIp: '10.0.0.3',   port: 8080, protocol: 'TCP',  bytes: 72000, reconstructionError: 6.4421, prediction: 1, predictionLabel: 'anomaly', severity: 'critical', timestamp: '2026-05-28T08:08:00Z' },
  { id: 'f010', rowIndex: 9,  srcIp: '192.168.1.15',  dstIp: '8.8.4.4',    port: 53,   protocol: 'UDP',  bytes: 88,    reconstructionError: 0.1908, prediction: 0, predictionLabel: 'normal',  severity: 'low',      timestamp: '2026-05-28T08:09:00Z' },
  { id: 'f011', rowIndex: 10, srcIp: '172.16.0.44',   dstIp: '10.0.0.1',   port: 23,   protocol: 'TCP',  bytes: 34000, reconstructionError: 3.5512, prediction: 1, predictionLabel: 'anomaly', severity: 'medium',   timestamp: '2026-05-28T08:10:00Z' },
  { id: 'f012', rowIndex: 11, srcIp: '192.168.1.16',  dstIp: '10.0.0.5',   port: 80,   protocol: 'TCP',  bytes: 1100,  reconstructionError: 0.6034, prediction: 0, predictionLabel: 'normal',  severity: 'low',      timestamp: '2026-05-28T08:11:00Z' },
  { id: 'f013', rowIndex: 12, srcIp: '10.10.0.202',   dstIp: '10.0.0.2',   port: 4444, protocol: 'TCP',  bytes: 61000, reconstructionError: 7.1290, prediction: 1, predictionLabel: 'anomaly', severity: 'critical', timestamp: '2026-05-28T08:12:00Z' },
  { id: 'f014', rowIndex: 13, srcIp: '192.168.1.17',  dstIp: '10.0.0.5',   port: 443,  protocol: 'TCP',  bytes: 2100,  reconstructionError: 0.3318, prediction: 0, predictionLabel: 'normal',  severity: 'low',      timestamp: '2026-05-28T08:13:00Z' },
]

export type InsightMock = {
  title: string
  body: string
  severity: 'low' | 'medium' | 'high'
}

export const insights: InsightMock[] = [
  {
    title: 'Phát hiện mẫu quét cổng',
    body: '3 luồng từ 10.10.0.200 nhắm mục tiêu các cổng 22, 23, 445 trong vòng 60 giây. Lỗi tái tạo vượt quá 5.5.',
    severity: 'high',
  },
  {
    title: 'Di chuyển nội bộ gia tăng',
    body: 'Máy chủ nội bộ 172.16.0.88 truy cập RDP (3389) trên 3 điểm cuối khác nhau. Hãy xác minh quyền truy cập.',
    severity: 'medium',
  },
  {
    title: 'Baseline bình thường ổn định',
    body: '115 luồng được phân loại bình thường. Lỗi tái tạo trung bình 0.38, cách xa ngưỡng 3.22.',
    severity: 'low',
  },
]

export type HealthMock = {
  label: string
  status: 'online' | 'degraded' | 'offline'
  uptime: string
  latency: string
}

export const health: HealthMock[] = [
  { label: 'Mô hình VAE',          status: 'online',   uptime: '99.9%',  latency: '12 ms'  },
  { label: 'Tiền xử lý',           status: 'online',   uptime: '99.9%',  latency: '4 ms'   },
  { label: 'Công cụ ngưỡng',      status: 'online',   uptime: '100%',   latency: '1 ms'   },
  { label: 'FastAPI Backend',    status: 'online',   uptime: '98.7%',  latency: '38 ms'  },
  { label: 'Scaler đặc trưng',     status: 'online',   uptime: '99.9%',  latency: '2 ms'   },
  { label: 'Bộ nạp PCAP',          status: 'degraded', uptime: '82.1%',  latency: '210 ms' },
]
