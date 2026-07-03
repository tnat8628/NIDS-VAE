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
  upload_id: string
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

/** Compact response: prediction đã lưu, không chứa toàn bộ flow results. */
export interface PredictionRunResponse {
  status: string
  upload_id: string
  inference_run_id: string
  summary: PredictionSummary
  results_url: string
}

export type Severity = 'low' | 'medium' | 'high' | 'critical'

export interface StoredFlowPrediction extends FlowPrediction {
  severity: Severity
}

export interface PaginationResponse {
  page: number
  page_size: number
  total_items: number
  total_pages: number
  has_previous: boolean
  has_next: boolean
}

export interface HistogramBinResponse {
  bin: string
  normal: number
  anomaly: number
}

export interface PredictionAggregatesResponse {
  histogram: HistogramBinResponse[]
  top_anomalies: StoredFlowPrediction[]
}

export interface PaginatedPredictionResponse {
  status: string
  upload_id: string
  inference_run_id: string
  summary: PredictionSummary
  items: StoredFlowPrediction[]
  pagination: PaginationResponse
  aggregates: PredictionAggregatesResponse
}

export interface DashboardOverviewResponse {
  status: string
  uploads: {
    total_uploads: number
    total_uploaded_flows: number
  }
  analysis: {
    analyzed_uploads: number
    total_analyzed_flows: number
    anomaly_count: number
    normal_count: number
    anomaly_rate: number
  }
  histogram: HistogramBinResponse[]
  classification: {
    normal: number
    anomaly: number
  }
  latest_activity: {
    latest_upload_id: string | null
    latest_run_id: string | null
    latest_filename: string | null
    latest_uploaded_at: string | null
    latest_predicted_at: string | null
  }
}

export type UploadAnalysisStatus = 'pending' | 'completed'

export interface UploadListItem {
  upload_id: string
  filename: string
  row_count: number
  col_count: number
  created_at: string
  analysis_status: UploadAnalysisStatus
  latest_run_id: string | null
  latest_predicted_at: string | null
  anomaly_count: number
  normal_count: number
}

export interface UploadListResponse {
  status: string
  items: UploadListItem[]
  pagination: PaginationResponse
}

export interface DeleteUploadResponse {
  status: string
  upload_id: string
  message: string
}

export type UploadFilter = 'all' | 'analyzed' | 'pending'
export type GlobalPredictionFilter = 'all' | 'anomaly' | 'normal'

export interface GlobalFlowItem {
  upload_id: string
  filename: string
  run_id: string
  row_index: number
  reconstruction_error: number
  prediction: 0 | 1
  prediction_label: 'normal' | 'anomaly'
  created_at: string
}

export interface GlobalFlowListResponse {
  status: string
  items: GlobalFlowItem[]
  pagination: PaginationResponse
}