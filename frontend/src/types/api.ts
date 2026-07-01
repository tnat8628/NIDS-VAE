/**
 * Định nghĩa kiểu TypeScript cho tất cả phản hồi API backend.
 * Nguồn sự thật: docs/api-spec.md
 */

// -------------------------------------------------------
// GET /health
// -------------------------------------------------------
export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error'
  model_loaded: boolean
  scaler_loaded: boolean
  threshold_loaded: boolean
  service_name: string
}

// -------------------------------------------------------
// POST /upload
// -------------------------------------------------------
export interface UploadResponse {
  status: string
  filename: string
  row_count: number
  col_count: number
  message: string
}

// -------------------------------------------------------
// POST /predict
// -------------------------------------------------------

/** Tóm tắt tổng quan kết quả dự đoán */
export interface PredictionSummary {
  total_flows: number
  anomaly_count: number
  normal_count: number
  anomaly_rate: number
  threshold: number
}

/** Kết quả dự đoán cho từng luồng mạng */
export interface FlowPrediction {
  row_index: number
  reconstruction_error: number
  prediction: 0 | 1
  prediction_label: 'normal' | 'anomaly'
}

/** Toàn bộ phản hồi từ POST /predict */
export interface PredictionResponse {
  status: string
  summary: PredictionSummary
  results: FlowPrediction[]
}
