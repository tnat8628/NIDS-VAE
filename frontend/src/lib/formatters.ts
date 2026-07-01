/**
 * Các hàm định dạng hiển thị dữ liệu cho dashboard.
 */

/** Định dạng số thực thành chuỗi với số chữ số thập phân cố định */
export function formatDecimal(value: number, digits = 4): string {
  return value.toFixed(digits)
}

/** Định dạng tỷ lệ (0–1) thành phần trăm: 0.1015 → "10.15%" */
export function formatPercent(value: number, digits = 2): string {
  return `${(value * 100).toFixed(digits)}%`
}

/** Định dạng số nguyên có dấu phân cách nghìn */
export function formatCount(value: number): string {
  return value.toLocaleString('vi-VN')
}

/** Trả về nhãn hiển thị tiếng Việt cho nhãn dự đoán backend */
export function formatPredictionLabel(label: 'normal' | 'anomaly'): string {
  return label === 'anomaly' ? 'Bất thường' : 'Bình thường'
}
