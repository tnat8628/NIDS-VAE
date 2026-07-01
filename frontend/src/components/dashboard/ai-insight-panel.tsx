import { useMemo } from "react";
import { Sparkles, Zap } from "lucide-react";
import type { MappedFlow, MappedSummary } from "@/lib/mapper";

const tone: Record<string, string> = {
  high: "border-anomaly/40 from-anomaly/10",
  medium: "border-violet/40 from-violet/10",
  low: "border-cyan/40 from-cyan/10",
};

interface AIInsightPanelProps {
  /** Summary và alert thật từ lần dự đoán gần nhất. */
  summary: MappedSummary;
  alertFlows: MappedFlow[];
}

export function AIInsightPanel({ summary, alertFlows }: AIInsightPanelProps) {
  // Tạo insight heuristic từ dữ liệu thật để tránh hiển thị nhận định mock.
  const insights = useMemo(() => {
    const topAlert = alertFlows[0];
    const anomalyRateTone = summary.anomalyRatePercent >= 20 ? "high" : summary.anomalyRatePercent >= 5 ? "medium" : "low";

    return [
      {
        title: "Tỷ lệ bất thường",
        severity: anomalyRateTone,
        body: `${summary.anomalyRatePercent.toFixed(2)}% flow vượt ngưỡng ${summary.threshold.toFixed(4)} trong batch mới nhất.`,
      },
      {
        title: "Lỗi tái tạo cao nhất",
        severity: topAlert?.severity === "critical" || topAlert?.severity === "high" ? "high" : "medium",
        body: topAlert
          ? `Row #${topAlert.rowIndex} có lỗi ${topAlert.reconstructionError.toFixed(4)}, thuộc nhóm ${topAlert.severity}.`
          : "Không có flow bất thường nào trong dữ liệu phân tích hiện tại.",
      },
      {
        title: "Phạm vi hiển thị",
        severity: summary.anomalyCount > alertFlows.length ? "medium" : "low",
        body: `Bảng cảnh báo đang hiển thị ${alertFlows.length.toLocaleString()} anomaly ưu tiên từ ${summary.anomalyCount.toLocaleString()} anomaly.`,
      },
    ];
  }, [alertFlows, summary]);

  return (
    <div className="relative rounded-xl p-px bg-gradient-primary shadow-glow">
      <div className="rounded-[11px] bg-card p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2.5">
            <div className="relative h-8 w-8 rounded-lg bg-gradient-primary grid place-items-center">
              <Sparkles className="h-4 w-4 text-primary-foreground" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold tracking-tight">Công cụ thông tin AI</h3>
                  <span className="text-[9px] uppercase tracking-[0.16em] px-1.5 py-0.5 rounded-full bg-gradient-primary text-primary-foreground font-semibold">Beta</span>
                </div>
              <p className="text-xs text-muted-foreground">Heuristic dựa trên quy tắc · Tích hợp LLM sắp ra mắt</p>
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] text-success">
            <span className="relative inline-flex h-2 w-2 rounded-full bg-success pulse-dot text-success" />
            <span className="uppercase tracking-wider">Trực tiếp</span>
          </div>
        </div>

        <div className="space-y-2.5">
          {insights.map((i) => (
            <div
              key={i.title}
              className={`relative rounded-lg border bg-gradient-to-r to-transparent p-3 ${tone[i.severity]}`}
            >
              <div className="flex items-start gap-2.5">
                <Zap className="h-3.5 w-3.5 mt-0.5 text-primary shrink-0" />
                <div className="min-w-0">
                  <div className="text-sm font-medium">{i.title}</div>
                  <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{i.body}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
