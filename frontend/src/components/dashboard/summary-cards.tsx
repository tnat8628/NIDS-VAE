import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  FileSpreadsheet,
  Gauge,
  Percent,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router-dom";
import type { MappedSummary } from "@/lib/mapper";
import type { DashboardOverviewResponse } from "@/types/api";

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

export function OverviewSummaryCards({
  overview,
}: {
  overview: DashboardOverviewResponse;
}) {
  const { uploads, analysis } = overview;
  const cards = [
    {
      label: "Số file CSV đã tải lên",
      value: uploads.total_uploads.toLocaleString(),
      delta: `${analysis.analyzed_uploads} file đã phân tích`,
      icon: FileSpreadsheet,
      tone: "cyan",
      to: "/uploads",
    },
    {
      label: "Tổng flow đã tải lên",
      value: uploads.total_uploaded_flows.toLocaleString(),
      delta: "từ toàn bộ file CSV",
      icon: Activity,
      tone: "cyan",
      to: "/uploads",
    },
    {
      label: "Flow đã phân tích",
      value: analysis.total_analyzed_flows.toLocaleString(),
      delta: `${analysis.total_analyzed_flows.toLocaleString()} / ${uploads.total_uploaded_flows.toLocaleString()} flow`,
      icon: Sparkles,
      tone: "success",
      to: "/uploads?filter=analyzed",
    },
    {
      label: "Bất thường đã phát hiện",
      value: analysis.anomaly_count.toLocaleString(),
      delta: "trên flow đã phân tích",
      icon: AlertTriangle,
      tone: "anomaly",
      to: "/flows?prediction=anomaly",
    },
    {
      label: "Flow bình thường",
      value: analysis.normal_count.toLocaleString(),
      delta: "trên flow đã phân tích",
      icon: CheckCircle2,
      tone: "success",
      to: "/flows?prediction=normal",
    },
    {
      label: "Tỷ lệ anomaly hệ thống",
      value: `${(analysis.anomaly_rate * 100).toFixed(2)}%`,
      delta: "trên tổng flow đã phân tích",
      icon: Percent,
      tone: "violet",
      to: "/flows",
    },
  ] as const;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <Link
            key={card.label}
            to={card.to}
            className="group relative overflow-hidden rounded-xl border border-border bg-card p-4 shadow-soft transition hover:-translate-y-0.5 hover:border-primary/50 hover:shadow-glow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring cursor-pointer"
          >
            <div
              className={`absolute -top-8 -right-8 h-24 w-24 rounded-full blur-2xl opacity-60 bg-gradient-to-br ${toneMap[card.tone]}`}
            />
            <div className="relative flex items-start justify-between">
              <div className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground truncate min-w-0 mr-2">
                {card.label}
              </div>
              <Icon className={`h-4 w-4 ${toneMap[card.tone].split(" ").pop()}`} />
            </div>
            <div className="relative mt-3 text-2xl font-semibold tracking-tight font-mono">
              {card.value}
            </div>
            <div className="relative mt-1 flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
              <span>{card.delta}</span>
              <span className="inline-flex items-center gap-1 text-primary opacity-80 transition group-hover:translate-x-0.5 group-hover:opacity-100">
                Xem chi tiết <ArrowUpRight className="h-3 w-3" />
              </span>
            </div>
          </Link>
        );
      })}
    </div>
  );
}