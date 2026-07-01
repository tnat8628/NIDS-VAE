import { useMemo } from "react";
import { ArrowUpRight } from "lucide-react";
import { MAX_ALERT_ROWS, type MappedFlow } from "@/lib/mapper";

interface AlertTableProps {
  /** Danh sach top anomaly da duoc gioi han, toi da 100 row. */
  flows: MappedFlow[];
}

const sevTone: Record<string, string> = {
  critical: "text-anomaly bg-anomaly/10 border-anomaly/30",
  high: "text-warning bg-warning/10 border-warning/30",
  medium: "text-violet bg-violet/10 border-violet/30",
  low: "text-cyan bg-cyan/10 border-cyan/30",
};

const sevLabel: Record<string, string> = {
  critical: "nghiêm trọng",
  high: "cao",
  medium: "trung bình",
  low: "thấp",
};

export function AlertTable({ flows }: AlertTableProps) {
  // Sap xep lai tren mang nho da cat gioi han; khong filter tat ca anomaly trong component.
  const alerts = useMemo(
    () => [...flows]
      .filter((flow) => flow.prediction === 1)
      .sort((a, b) => b.reconstructionError - a.reconstructionError)
      .slice(0, MAX_ALERT_ROWS),
    [flows],
  );

  return (
    <div className="rounded-xl border border-border bg-card shadow-soft">
      <div className="flex items-center justify-between p-5 pb-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">Cảnh báo gần đây</h3>
          <p className="text-xs text-muted-foreground">Top anomaly theo lỗi tái tạo, tối đa {MAX_ALERT_ROWS} dòng</p>
        </div>
        <button className="text-xs text-primary hover:underline flex items-center gap-1">Xem tất cả <ArrowUpRight className="h-3 w-3" /></button>
      </div>
      <div className="overflow-hidden">
        <table className="w-full text-sm">
          <thead className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground bg-muted/30 border-y border-border">
            <tr>
              <th className="text-left font-medium py-2 px-5">Hàng #</th>
              <th className="text-left font-medium py-2 px-3">Lỗi tái tạo</th>
              <th className="text-left font-medium py-2 px-3">Nhãn</th>
              <th className="text-left font-medium py-2 px-3">Mức độ</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((alert) => (
              <tr key={alert.rowIndex} className="border-b border-border last:border-0 hover:bg-muted/30 transition">
                <td className="py-2.5 px-5 font-mono text-xs text-muted-foreground">{alert.rowIndex}</td>
                <td className="py-2.5 px-3 font-mono text-xs text-anomaly">{alert.reconstructionError.toFixed(4)}</td>
                <td className="py-2.5 px-3">
                  <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full border text-anomaly bg-anomaly/10 border-anomaly/30">
                    Bất thường
                  </span>
                </td>
                <td className="py-2.5 px-3">
                  <span className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full border ${sevTone[alert.severity]}`}>
                    {sevLabel[alert.severity]}
                  </span>
                </td>
              </tr>
            ))}
            {alerts.length === 0 && (
              <tr>
                <td colSpan={4} className="py-8 text-center text-xs text-muted-foreground">
                  Không có bất thường nào được phát hiện
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
