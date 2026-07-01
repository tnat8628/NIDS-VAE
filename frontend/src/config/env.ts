/**
 * Biến môi trường – đọc từ .env (VITE_ prefix bắt buộc với Vite).
 * Tạo file .env.local để override khi phát triển local.
 *
 * Ví dụ .env.local:
 *   VITE_API_BASE_URL=http://127.0.0.1:8000
 */
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export const APP_ENV: string =
  import.meta.env.MODE ?? 'development'

export const IS_DEV = APP_ENV === 'development'
