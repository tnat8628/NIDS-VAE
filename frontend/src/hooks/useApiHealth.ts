import { useState, useEffect } from 'react'
import { getHealth } from '@/lib/api'
import type { HealthResponse } from '@/types/api'

interface UseApiHealthResult {
  health: HealthResponse | null
  loading: boolean
  error: string | null
  refetch: () => void
}

/**
 * Hook kiểm tra trạng thái sức khoẻ của backend (GET /health).
 * Tự động gọi khi component mount.
 */
export function useApiHealth(): UseApiHealthResult {
  const [health, setHealth]   = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)
  const [tick, setTick]       = useState(0)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    getHealth()
      .then((data) => {
        if (!cancelled) setHealth(data)
      })
      .catch((err) => {
        if (!cancelled)
          setError(err?.response?.data?.detail ?? err.message ?? 'Lỗi kết nối backend')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => { cancelled = true }
  }, [tick])

  const refetch = () => setTick((t) => t + 1)

  return { health, loading, error, refetch }
}
