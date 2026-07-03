import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowUpRight, Database, Upload } from "lucide-react";
import { Topbar } from "@/components/common/topbar";
import { OverviewSummaryCards } from "@/components/dashboard/summary-cards";
import { ErrorHistogram } from "@/components/dashboard/error-histogram";
import { AnomalyDonut } from "@/components/dashboard/anomaly-donut";
import { getDashboardOverview } from "@/lib/api";
import type { MappedSummary } from "@/lib/mapper";
import type { DashboardOverviewResponse } from "@/types/api";

function formatDate(value: string | null): string {
  if (!value) return "Chưa có";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Chưa có" : date.toLocaleString("vi-VN");
}

export default function DashboardPage() {
  const [overview, setOverview] = useState<DashboardOverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDashboardOverview()
      .then(setOverview)
      .catch(() => setError("Không thể tải thống kê tổng hợp từ PostgreSQL."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <>
        <Topbar title="Tổng quan" subtitle="Đang tổng hợp dữ liệu từ PostgreSQL..." />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-sm text-muted-foreground">Đang tải...</div>
        </main>
      </>
    );
  }

  if (!overview) {
    return (
      <>
        <Topbar title="Tổng quan" subtitle="Không thể tải dữ liệu" />
        <main className="flex-1 flex items-center justify-center px-4">
          <div className="text-center space-y-4 max-w-sm">
            <p className="text-sm text-muted-foreground">{error}</p>
            <Link
              to="/upload"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium"
            >
              <Upload className="h-4 w-4" />
              Tải lên file CSV
            </Link>
          </div>
        </main>
      </>
    );
  }

  const analyzedSummary: MappedSummary = {
    totalFlows: overview.analysis.total_analyzed_flows,
    anomalyCount: overview.classification.anomaly,
    normalCount: overview.classification.normal,
    anomalyRatePercent: overview.analysis.anomaly_rate * 100,
    threshold: 0,
  };
  const pendingFlows = Math.max(
    0,
    overview.uploads.total_uploaded_flows - overview.analysis.total_analyzed_flows,
  );
  const latest = overview.latest_activity;
  const latestResultsUrl =
    latest.latest_upload_id && latest.latest_run_id
      ? `/results?uploadId=${encodeURIComponent(latest.latest_upload_id)}&runId=${encodeURIComponent(latest.latest_run_id)}`
      : null;

  return (
    <>
      <Topbar
        title="Tổng quan"
        subtitle="Thống kê tổng hợp từ toàn bộ dữ liệu đã tải lên"
      />
      <main className="flex-1 w-full px-4 md:px-6 lg:px-8 py-4 md:py-6 space-y-4 lg:space-y-6">
        <OverviewSummaryCards overview={overview} />

        <div className="grid grid-cols-12 gap-4 lg:gap-6">
          <div className="col-span-12 xl:col-span-8 min-w-0 overflow-hidden">
            <ErrorHistogram histogram={overview.histogram} />
          </div>
          <div className="col-span-12 xl:col-span-4 min-w-0 overflow-hidden">
            <AnomalyDonut
              summary={analyzedSummary}
              subtitle="Latest inference run của mỗi file"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 lg:gap-6">
          <div className="rounded-xl border border-border bg-card p-5 shadow-soft">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-semibold">Phạm vi phân tích</h3>
            </div>
            <p className="mt-2 text-2xl font-mono font-semibold">
              {overview.analysis.total_analyzed_flows.toLocaleString()} /{" "}
              {overview.uploads.total_uploaded_flows.toLocaleString()} flow
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {pendingFlows.toLocaleString()} flow đã tải lên nhưng chưa có kết quả VAE.
              Anomaly và normal chỉ tính trên flow đã phân tích.
            </p>
          </div>

          <div className="rounded-xl border border-border bg-card p-5 shadow-soft">
            <h3 className="text-sm font-semibold">Hoạt động phân tích mới nhất</h3>
            <div className="mt-3 space-y-1 text-xs text-muted-foreground">
              <p>File: <span className="text-foreground">{latest.latest_filename ?? "Chưa có"}</span></p>
              <p>Tải lên: <span className="text-foreground">{formatDate(latest.latest_uploaded_at)}</span></p>
              <p>Phân tích: <span className="text-foreground">{formatDate(latest.latest_predicted_at)}</span></p>
            </div>
            {latestResultsUrl && (
              <Link
                to={latestResultsUrl}
                className="mt-3 inline-flex items-center gap-1 text-xs text-primary hover:underline"
              >
                Mở kết quả file này <ArrowUpRight className="h-3 w-3" />
              </Link>
            )}
          </div>
        </div>
      </main>
    </>
  );
}