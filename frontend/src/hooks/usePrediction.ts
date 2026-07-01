import { useState, useCallback } from 'react'
import { predictCsv } from '@/lib/api'
import type { PredictionResponse } from '@/types/api'

interface UsePredictionResult {
  result: PredictionResponse | null
  loading: boolean
  error: string | null
  predict: (file: File) => Promise<void>
  reset: () => void
}

/**
 * Hook chạy pipeline dự đoán VAE (POST /predict).
 * Lưu kết quả trong state để component con có thể đọc.
 */
export function usePrediction(): UsePredictionResult {
  const [result, setResult]   = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const predict = useCallback(async (file: File) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await predictCsv(file)
      setResult(data)
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } }; message?: string }
      setError(
        axiosError?.response?.data?.detail ??
        axiosError?.message ??
        'Dự đoán thất bại. Kiểm tra kết nối backend.'
      )
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  return { result, loading, error, predict, reset }
}
