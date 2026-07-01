import { useMemo } from "react";
import { Link } from "react-router-dom";
import { Upload } from "lucide-react";
import { Topbar } from "@/components/common/topbar";
import { SummaryCards } from "@/components/dashboard/summary-cards";
import { ErrorHistogram } from "@/components/dashboard/error-histogram";
import { AnomalyDonut } from "@/components/dashboard/anomaly-donut";
import { SeverityBar } from "@/components/dashboard/severity-bar";
import { ActivityTimeline } from "@/components/dashboard/activity-timeline";
import { AlertTable } from "@/components/dashboard/alert-table";
import { loadPredictionFromStorage } from "@/lib/mapper";

const formatPredictionTime = (createdAt?: string) => {
  if (!createdAt) return "Dữ liệu dự đoán mới nhất";

  const date = new Date(createdAt);
  if (Number.isNaN(date.getTime())) return "Dữ liệu dự đoán mới nhất";

  return `Lần phân tích mới nhất: ${date.toLocaleString("vi-VN")}`;
};

export default function DashboardPage() {
  // Dashboard chỉ đọc cache nhẹ đã được UploadPage lưu sau khi gọi predictCsv().
  const prediction = useMemo(() => loadPredictionFromStorage(), []);

  if (!prediction) {
    return (
      <>
        <Topbar title="Bảng điều khiển" subtitle="Chưa có dữ liệu phân tích" />
        <main className="flex-1 flex items-center justify-center px-4">
          <div className="text-center space-y-4 max-w-sm">
            <div>
              <h2 className="text-lg font-semibold tracking-tight">Chưa có dữ liệu phân tích.</h2>
              <p className="mt-2 text-sm text-muted-foreground">Vui lòng tải lên file CSV để bắt đầu.</p>
            </div>
            <Link
              to="/upload"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition"
            >
              <Upload className="h-4 w-4" />
              Tải lên file CSV
            </Link>
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <Topbar title="Bảng điều khiển" subtitle={`Kết quả VAE từ localStorage · ${formatPredictionTime(prediction.createdAt)}`} />
      <main className="flex-1 w-full px-4 md:px-6 lg:px-8 py-4 md:py-6 space-y-4 lg:space-y-6">
        <SummaryCards summary={prediction.summary} />
        <div className="grid grid-cols-12 gap-4 lg:gap-6">
          <div className="col-span-12 xl:col-span-8 min-w-0 overflow-hidden"><ErrorHistogram histogram={prediction.histogram} threshold={prediction.summary.threshold} /></div>
          <div className="col-span-12 xl:col-span-4 min-w-0 overflow-hidden"><AnomalyDonut summary={prediction.summary} /></div>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 lg:gap-6">
          <div className="xl:col-span-2 min-w-0 overflow-hidden"><ActivityTimeline flows={prediction.flows} /></div>
          <div className="min-w-0 overflow-hidden"><SeverityBar flows={prediction.alertFlows} /></div>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 lg:gap-6">
          <div className="xl:col-span-2 min-w-0 overflow-hidden"><AlertTable flows={prediction.alertFlows} /></div>
          {/* <div className="min-w-0 overflow-hidden"><AIInsightPanel summary={prediction.summary} alertFlows={prediction.alertFlows} /></div> */}
        </div>
      </main>
    </>
  );
}
