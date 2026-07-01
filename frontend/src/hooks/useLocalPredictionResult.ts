import { useCallback, useState } from 'react'
import {
  PREDICTION_STORAGE_KEY,
  buildLocalStorageCache,
  loadPredictionFromStorage,
  type LightweightPredictionCache,
} from '@/lib/mapper'
import type { PredictionResponse } from '@/types/api'

/**
 * Hook luu/doc ket qua du doan dang cache nhe.
 * Chi giu summary, histogram va cac row preview de tranh day localStorage bang payload lon.
 */
export function useLocalPredictionResult() {
  const [result, setResultState] = useState<LightweightPredictionCache | null>(() => loadPredictionFromStorage())

  /** Chuyen raw /predict response thanh cache nhe truoc khi luu vao state va localStorage. */
  const saveResult = useCallback((data: PredictionResponse) => {
    const cache = buildLocalStorageCache(data)
    setResultState(cache)
    try {
      localStorage.setItem(PREDICTION_STORAGE_KEY, JSON.stringify(cache))
    } catch {
      // localStorage co the day quota; bo qua de UI khong crash sau khi /predict thanh cong.
    }
  }, [])

  /** Xoa ket qua cache khoi state va localStorage. */
  const clearResult = useCallback(() => {
    setResultState(null)
    localStorage.removeItem(PREDICTION_STORAGE_KEY)
  }, [])

  return { result, saveResult, clearResult }
}
