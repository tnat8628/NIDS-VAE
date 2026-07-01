/**
 * mapper.ts
 * Chuyen doi du lieu tho tu backend API sang view model gon nhe cho UI.
 * Muc tieu chinh: khong giu toan bo response.results lon trong React state/localStorage.
 */

import type { FlowPrediction, PredictionResponse, PredictionSummary } from '@/types/api'

export const MAX_PREVIEW_ROWS = 500
export const MAX_ALERT_ROWS = 100
export const MAX_LOCAL_STORAGE_ROWS = 100
export const HISTOGRAM_BINS = 30

/** Muc do nghiem trong duoc tinh tu reconstruction error va threshold. */
export type Severity = 'low' | 'medium' | 'high' | 'critical'

/** Thong tin mot flow sau khi anh xa sang du lieu hien thi. */
export interface MappedFlow {
  rowIndex: number
  reconstructionError: number
  prediction: 0 | 1
  predictionLabel: 'normal' | 'anomaly'
  severity: Severity
}

/** Tom tat ket qua phan tich sau khi anh xa. */
export interface MappedSummary {
  totalFlows: number
  anomalyCount: number
  normalCount: number
  /** Ty le bat thuong theo phan tram (0-100). */
  anomalyRatePercent: number
  threshold: number
}

/** Bin histogram cho bieu do phan bo loi. */
export interface HistogramBin {
  bin: string
  normal: number
  anomaly: number
}

/** View model nhe RAM ma dashboard co the giu trong React state an toan. */
export interface MappedPrediction {
  summary: MappedSummary
  histogram: HistogramBin[]
  flows: MappedFlow[]
  alertFlows: MappedFlow[]
  isPreview: boolean
  originalResultCount: number
}

/** Cache localStorage chi gom metadata va cac mang da cat gioi han. */
export type LightweightPredictionCache = MappedPrediction & {
  createdAt: string
}

/** Tinh severity dua tren reconstruction error so voi threshold. */
export function computeSeverity(error: number, threshold: number): Severity {
  if (error > threshold * 2) return 'critical'
  if (error > threshold * 1.5) return 'high'
  if (error > threshold) return 'medium'
  return 'low'
}

/** Anh xa mot flow API sang flow UI, giu logic threshold dong nhat voi backend. */
function mapFlow(flow: FlowPrediction, threshold: number): MappedFlow {
  return {
    rowIndex: flow.row_index,
    reconstructionError: flow.reconstruction_error,
    prediction: flow.prediction,
    predictionLabel: flow.prediction_label,
    severity: computeSeverity(flow.reconstruction_error, threshold),
  }
}

/** Anh xa summary API sang summary UI. */
function mapSummary(summary: PredictionSummary): MappedSummary {
  return {
    totalFlows: summary.total_flows,
    anomalyCount: summary.anomaly_count,
    normalCount: summary.normal_count,
    anomalyRatePercent: summary.anomaly_rate * 100,
    threshold: summary.threshold,
  }
}

/** Tao 30 bin co dinh de chart khong can nhan mang errors day du. */
function createHistogramBins(minError: number, maxError: number): HistogramBin[] {
  if (!Number.isFinite(minError) || !Number.isFinite(maxError)) return []

  const safeMin = Math.min(minError, maxError)
  const safeMax = Math.max(minError, maxError)
  const range = safeMax - safeMin
  const step = range > 0 ? range / HISTOGRAM_BINS : 1

  return Array.from({ length: HISTOGRAM_BINS }, (_, index) => ({
    bin: (safeMin + index * step).toFixed(4),
    normal: 0,
    anomaly: 0,
  }))
}

/** Tim vi tri bin cua mot error ma khong tao them mang trung gian. */
function getHistogramIndex(error: number, minError: number, maxError: number): number {
  if (maxError <= minError) return 0
  const ratio = (error - minError) / (maxError - minError)
  return Math.min(HISTOGRAM_BINS - 1, Math.max(0, Math.floor(ratio * HISTOGRAM_BINS)))
}

/** Chen anomaly vao danh sach top-N da sap xep giam dan theo reconstruction_error. */
function insertTopAlert(alerts: MappedFlow[], flow: MappedFlow, limit: number): void {
  const insertAt = alerts.findIndex((item) => item.reconstructionError < flow.reconstructionError)
  if (insertAt === -1) {
    if (alerts.length < limit) alerts.push(flow)
    return
  }

  alerts.splice(insertAt, 0, flow)
  if (alerts.length > limit) alerts.pop()
}

/**
 * Tao dashboard view model gon nhe tu PredictionResponse.
 * Chi giu summary, histogram bins, top anomaly va preview flow de tranh Out of Memory tren Chrome.
 */
export function buildDashboardViewModel(
  response: PredictionResponse,
  options: { previewLimit?: number; alertLimit?: number } = {},
): MappedPrediction {
  const threshold = response.summary?.threshold ?? 0
  const results = response.results ?? []
  const previewLimit = options.previewLimit ?? MAX_PREVIEW_ROWS
  const alertLimit = options.alertLimit ?? MAX_ALERT_ROWS
  const previewFlows: MappedFlow[] = []
  const alertFlows: MappedFlow[] = []

  // Luot dau chi tim min/max de chia bin, khong tao mang errors lon gay ton RAM.
  let minError = Number.POSITIVE_INFINITY
  let maxError = Number.NEGATIVE_INFINITY
  for (const flow of results) {
    const error = flow.reconstruction_error
    if (!Number.isFinite(error)) continue
    minError = Math.min(minError, error)
    maxError = Math.max(maxError, error)
  }

  const histogram = createHistogramBins(minError, maxError)

  for (const flow of results) {
    const mapped = mapFlow(flow, threshold)

    // Chi giu ban xem truoc huu han de React khong phai giu/render hang nghin hoac hang trieu row.
    if (previewFlows.length < previewLimit) previewFlows.push(mapped)

    // Bang canh bao chi can top anomaly theo reconstruction_error, khong giu tat ca anomaly.
    if (mapped.prediction === 1) insertTopAlert(alertFlows, mapped, alertLimit)

    if (histogram.length > 0 && Number.isFinite(mapped.reconstructionError)) {
      const index = getHistogramIndex(mapped.reconstructionError, minError, maxError)
      if (mapped.prediction === 1) histogram[index].anomaly += 1
      else histogram[index].normal += 1
    }
  }

  return {
    summary: mapSummary(response.summary),
    histogram,
    flows: previewFlows,
    alertFlows,
    isPreview: results.length > previewFlows.length,
    originalResultCount: results.length,
  }
}

/** Ten cu duoc giu lai cho cac component hien co; ket qua da la view model nhe RAM. */
export function mapPredictionResponse(response: PredictionResponse): MappedPrediction {
  return buildDashboardViewModel(response)
}

/**
 * Tao histogram tu danh sach flow nho.
 * Ham nay chi nen dung cho mock/preview; ResultsPage dung histogram da tinh san trong view model.
 */
export function computeHistogram(flows: MappedFlow[], binSize = 0.5): HistogramBin[] {
  if (flows.length === 0) return []

  const maxErr = Math.max(...flows.map((flow) => flow.reconstructionError))
  const numBins = Math.min(HISTOGRAM_BINS, Math.ceil(maxErr / binSize) + 1)
  const bins: HistogramBin[] = Array.from({ length: numBins }, (_, index) => ({
    bin: (index * binSize).toFixed(1),
    normal: 0,
    anomaly: 0,
  }))

  for (const flow of flows) {
    const index = Math.min(Math.floor(flow.reconstructionError / binSize), numBins - 1)
    if (flow.prediction === 0) bins[index].normal += 1
    else bins[index].anomaly += 1
  }

  return bins
}

export const PREDICTION_STORAGE_KEY = 'nids_last_prediction'

/** Tao cache localStorage toi da 100 rows; tuyet doi khong stringify toan bo response.results. */
export function buildLocalStorageCache(response: PredictionResponse): LightweightPredictionCache {
  return {
    ...buildDashboardViewModel(response, {
      previewLimit: MAX_LOCAL_STORAGE_ROWS,
      alertLimit: MAX_LOCAL_STORAGE_ROWS,
    }),
    createdAt: new Date().toISOString(),
  }
}

function isLightweightPredictionCache(value: unknown): value is LightweightPredictionCache {
  const candidate = value as Partial<LightweightPredictionCache> | null
  return Boolean(candidate?.summary && Array.isArray(candidate.flows) && Array.isArray(candidate.histogram))
}

/** Luu ban cache gon nhe vao localStorage, tranh stringify payload lon lam day quota/RAM. */
export function savePredictionToStorage(response: PredictionResponse): void {
  try {
    localStorage.setItem(PREDICTION_STORAGE_KEY, JSON.stringify(buildLocalStorageCache(response)))
  } catch {
    // Neu localStorage day hoac bi chan, bo qua de man hinh ket qua van tiep tuc hoat dong.
  }
}

/** Doc cache nhe tu localStorage; khong nap lai payload results day du vao React state. */
export function loadPredictionFromStorage(): LightweightPredictionCache | null {
  try {
    const raw = localStorage.getItem(PREDICTION_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as unknown
    if (isLightweightPredictionCache(parsed)) return parsed

    // Xoa cache cu neu no van la PredictionResponse day du de tranh lap lai loi Out of Memory.
    localStorage.removeItem(PREDICTION_STORAGE_KEY)
    return null
  } catch {
    return null
  }
}
