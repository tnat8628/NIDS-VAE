import axios from 'axios'
import { API_BASE_URL } from '@/config/env'
import type {
  HealthResponse,
  UploadResponse,
  PredictionResponse,
} from '@/types/api'

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

/** Lấy kết quả dự đoán gần nhất từ server (GET /results) */
export async function getResults(): Promise<PredictionResponse> {
  const { data } = await apiClient.get<PredictionResponse>('/results')
  return data
}

export default apiClient
