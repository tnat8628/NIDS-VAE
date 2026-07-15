import axios from 'axios'
import { API_BASE_URL } from '@/config/env'
import type {
  DashboardOverviewResponse,
  DeleteUploadResponse,
  GlobalFlowListResponse,
  GlobalPredictionFilter,
  HealthResponse,
  PaginatedPredictionResponse,
  UploadFilter,
  UploadListResponse,
  UploadResponse,
  PredictionResponse,
  PredictionRunResponse,
} from '@/types/api'

export type ResultsCsvPredictionFilter = 'all' | 'anomaly' | 'normal'
export type ResultsCsvSort = 'idx' | 'err_desc' | 'err_asc'

/**
 * Instance Axios dùng chung cho toàn ứng dụng.
 * Base URL lấy từ biến môi trường VITE_API_BASE_URL,
 * fallback về http://127.0.0.1:8000 khi không có env.
 */
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60_000, // 60 giây – phù hợp cho inference file lớn
  headers: {
    Accept: 'application/json',
  },
})

// -------------------------------------------------------
// Response interceptor – log lỗi trong chế độ dev
// -------------------------------------------------------
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (import.meta.env.DEV) {
      console.error('[API Error]', error?.response?.data ?? error.message)
    }
    return Promise.reject(error)
  },
)

// -------------------------------------------------------
// API stub functions – sẽ được kết nối với backend thật
// -------------------------------------------------------

/** Kiểm tra trạng thái service và các artifact (GET /health) */
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>('/health')
  return data
}

/**
 * Tải lên file CSV để kiểm tra tính hợp lệ (POST /upload).
 * Không chạy inference – dùng để validate trước.
 */
export async function uploadCsv(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await apiClient.post<UploadResponse>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

/**
 * Chạy pipeline dự đoán VAE trên file CSV (POST /predict).
 * Trả về summary và kết quả từng luồng.
 */
export async function predictCsv(file: File): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await apiClient.post<PredictionResponse>('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

/** Aggregate toàn hệ thống, độc lập với upload/run đang mở ở ResultsPage. */
export async function getDashboardOverview(): Promise<DashboardOverviewResponse> {
  const { data } = await apiClient.get<DashboardOverviewResponse>('/dashboard/overview')
  return data
}

/** Danh sÃ¡ch CSV upload cÃ³ phÃ¢n trang vÃ  filter tá»« PostgreSQL. */
export async function getUploads(
  page = 1,
  pageSize = 20,
  filter: UploadFilter = 'all',
): Promise<UploadListResponse> {
  const { data } = await apiClient.get<UploadListResponse>('/uploads', {
    params: { page, page_size: pageSize, filter },
  })
  return data
}

/** XÃ³a má»™t upload vÃ  toÃ n bá»™ rows/runs/predictions liÃªn quan. */
export async function deleteUpload(uploadId: string): Promise<DeleteUploadResponse> {
  const { data } = await apiClient.delete<DeleteUploadResponse>(
    `/uploads/${encodeURIComponent(uploadId)}`,
  )
  return data
}

/** Flow explorer toÃ n há»‡ thá»‘ng, chá»‰ láº¥y latest run cá»§a má»—i upload. */
export async function getDashboardFlows(
  page = 1,
  pageSize = 25,
  prediction: GlobalPredictionFilter = 'all',
): Promise<GlobalFlowListResponse> {
  const { data } = await apiClient.get<GlobalFlowListResponse>('/dashboard/flows', {
    params: { page, page_size: pageSize, prediction },
  })
  return data
}

/** Chạy inference cho upload đã lưu; response không chứa mảng flow khổng lồ. */
export async function predictUpload(uploadId: string): Promise<PredictionRunResponse> {
  const { data } = await apiClient.post<PredictionRunResponse>(
    `/uploads/${encodeURIComponent(uploadId)}/predict`,
    undefined,
    { timeout: 300_000 },
  )
  return data
}

/** Lấy đúng một trang results từ PostgreSQL. */
export async function getUploadResults(
  uploadId: string,
  page = 1,
  pageSize = 25,
  inferenceRunId?: string,
): Promise<PaginatedPredictionResponse> {
  const { data } = await apiClient.get<PaginatedPredictionResponse>(
    `/uploads/${encodeURIComponent(uploadId)}/results`,
    {
      params: {
        page,
        page_size: pageSize,
        ...(inferenceRunId ? { inference_run_id: inferenceRunId } : {}),
      },
    },
  )
  return data
}

/** Tai file CSV day du cua mot inference run theo filter hien tai. */
export async function downloadUploadResultsCsv(
  uploadId: string,
  inferenceRunId: string | undefined,
  prediction: ResultsCsvPredictionFilter = 'all',
  sort: ResultsCsvSort = 'idx',
): Promise<{ blob: Blob; filename: string }> {
  const response = await apiClient.get<Blob>(
    `/uploads/${encodeURIComponent(uploadId)}/results/export`,
    {
      params: {
        prediction,
        sort,
        ...(inferenceRunId ? { inference_run_id: inferenceRunId } : {}),
      },
      responseType: 'blob',
      timeout: 300_000,
    },
  )

  const disposition = response.headers['content-disposition']
  const match =
    typeof disposition === 'string'
      ? disposition.match(/filename="?([^";]+)"?/i)
      : null
  const filename = match?.[1] ?? `nids_results_${uploadId}_${prediction}.csv`

  return { blob: response.data, filename }
}

/** Lấy kết quả dự đoán gần nhất từ server (GET /results) */
export async function getResults(): Promise<PredictionResponse> {
  const { data } = await apiClient.get<PredictionResponse>('/results')
  return data
}

export default apiClient