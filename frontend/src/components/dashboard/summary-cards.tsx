import { Activity, AlertTriangle, CheckCircle2, Gauge, Percent, Sparkles } from "lucide-react";
import type { MappedSummary } from "@/lib/mapper";

interface SummaryCardsProps {
  /** Dữ liệu tóm tắt từ kết quả dự đoán thật */
  summary: MappedSummary;
}

const toneMap: Record<string, string> = {
  cyan: "from-cyan/20 to-cyan/0 text-cyan",
  anomaly: "from-anomaly/20 to-anomaly/0 text-anomaly",
  success: "from-success/20 to-success/0 text-success",
  violet: "from-violet/20 to-violet/0 text-violet",
  warning: "from-warning/20 to-warning/0 text-warning",
};

export function SummaryCards({ summary }: SummaryCardsProps) {
  // Tạo danh sách card từ dữ liệu thật
  const cards = [
    { label: "Tổng luồng", value: summary.totalFlows.toLocaleString(), delta: "đã phân tích", icon: Activity, tone: "cyan" },
    { label: "Bất thường phát hiện", value: summary.anomalyCount.toString(), delta: "vượt ngưỡng", icon: AlertTriangle, tone: "anomaly" },
    { label: "Luồng bình thường", value: summary.normalCount.toString(), delta: "dưới ngưỡng", icon: CheckCircle2, tone: "success" },
    { label: "Tỷ lệ bất thường", value: `${summary.anomalyRatePercent.toFixed(2)}%`, delta: "của tổng luồng", icon: Percent, tone: "violet" },
    { label: "Ngưỡng", value: summary.threshold.toFixed(4), delta: "tự động từ training", icon: Gauge, tone: "warning" },
    { label: "Trạng thái mô hình", value: "Hoạt động", delta: "VAE NIDS", icon: Sparkles, tone: "success" },
  ] as const;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {cards.map((c) => {
        const Icon = c.icon;
        return (
          <div
            key={c.label}
            className="relative overflow-hidden rounded-xl border border-border bg-card p-4 shadow-soft hover:border-primary/40 transition group"
          >
            <div
              className={`absolute -top-8 -right-8 h-24 w-24 rounded-full blur-2xl opacity-60 bg-gradient-to-br ${toneMap[c.tone]}`}
            />
            <div className="relative flex items-start justify-between">
              <div className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground truncate min-w-0 mr-2">{c.label}</div>
              <Icon className={`h-4 w-4 ${toneMap[c.tone].split(" ").pop()}`} />
            </div>
            <div className="relative mt-3 text-2xl font-semibold tracking-tight font-mono">{c.value}</div>
            <div className="relative mt-1 text-[11px] text-muted-foreground">{c.delta}</div>
          </div>
        );
      })}
    </div>
  );
}
