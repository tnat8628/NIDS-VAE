/**
 * Map bounded database-backed API responses to existing dashboard view models.
 */

import type {
  PaginatedPredictionResponse,
  PredictionSummary,
  Severity,
  StoredFlowPrediction,
} from '@/types/api'

export const MAX_ALERT_ROWS = 100

export type { Severity }

export interface MappedFlow {
  rowIndex: number
  reconstructionError: number
  prediction: 0 | 1
  predictionLabel: 'normal' | 'anomaly'
  actualLabel: string | number | boolean | null
  actualBinary: 0 | 1 | null
  severity: Severity
}

export interface MappedSummary {
  totalFlows: number
  anomalyCount: number
  normalCount: number
  anomalyRatePercent: number
  threshold: number
}

export interface HistogramBin {
  bin: string
  normal: number
  anomaly: number
}

export interface MappedPrediction {
  summary: MappedSummary
  histogram: HistogramBin[]
  flows: MappedFlow[]
  alertFlows: MappedFlow[]
}

function mapFlow(flow: StoredFlowPrediction): MappedFlow {
  return {
    rowIndex: flow.row_index,
    reconstructionError: flow.reconstruction_error,
    prediction: flow.prediction,
    predictionLabel: flow.prediction_label,
    actualLabel: flow.actual_label,
    actualBinary: flow.actual_binary,
    severity: flow.severity,
  }
}

function mapSummary(summary: PredictionSummary): MappedSummary {
  return {
    totalFlows: summary.total_flows,
    anomalyCount: summary.anomaly_count,
    normalCount: summary.normal_count,
    anomalyRatePercent: summary.anomaly_rate * 100,
    threshold: summary.threshold,
  }
}

export function mapPaginatedResults(
  response: PaginatedPredictionResponse,
): MappedPrediction {
  return {
    summary: mapSummary(response.summary),
    histogram: response.aggregates.histogram,
    flows: response.items.map(mapFlow),
    alertFlows: response.aggregates.top_anomalies.map(mapFlow),
  }
}