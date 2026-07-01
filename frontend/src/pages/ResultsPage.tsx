import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Upload } from "lucide-react";
import { Topbar } from "@/components/common/topbar";
import { SummaryCards } from "@/components/dashboard/summary-cards";
import { ErrorHistogram } from "@/components/dashboard/error-histogram";
import { FlowTable } from "@/components/dashboard/flow-table";
import { AlertTable } from "@/components/dashboard/alert-table";
import { getResults } from "@/lib/api";
import {
  buildDashboardViewModel,
  loadPredictionFromStorage,
  type MappedPrediction,
} from "@/lib/mapper";
import type { PredictionResponse } from "@/types/api";

type ResultsRouteState = {
  predictionView?: MappedPrediction
  prediction?: PredictionResponse
}

const LARGE_RESULT_WARNING =
  "File có nhiều flow, hệ thống đang hiển thị bản tóm tắt để tránh quá tải trình duyệt.";

export default function ResultsPage() {
  // React state chi giu view model da cat gioi han, khong giu raw response.results.
  const [prediction, setPrediction] = useState<MappedPrediction | null>(null);
  const [loading, setLoading] = useState(true);
  const location = useLocation();

  useEffect(() => {
    async function loadPrediction() {
      const routeState = (location.state as ResultsRouteState | null) ?? null;

      // Uu tien 1: view model nhe tu UploadPage, hien thi ngay sau khi /predict xong.
      if (routeState?.predictionView) {
        setPrediction(routeState.predictionView);
        setLoading(false);
        return;
      }

      // Tuong thich nguoc: neu co raw PredictionResponse cu, chuyen ngay sang view model nhe.
      if (routeState?.prediction) {
        setPrediction(buildDashboardViewModel(routeState.prediction));
        setLoading(false);
        return;
      }

      // Uu tien 2: localStorage chi duoc phep chua cache nhe da gioi han so row.
      const stored = loadPredictionFromStorage();
      if (stored) {
        setPrediction(stored);
        setLoading(false);
        return;
      }

      // Fallback GET /results co the tra payload rat lon, nen chi bat khi URL co ?fetchLatest=1.
      const params = new URLSearchParams(location.search);
      if (params.get("fetchLatest") !== "1") {
        setLoading(false);
        return;
      }

      try {
        const response = await getResults();
        if (response?.results?.length) {
          setPrediction(buildDashboardViewModel(response));
        }
      } catch {
        // Khong co ket qua moi - hien thi trang rong thay vi crash.
      } finally {
        setLoading(false);
      }
    }

    loadPrediction();
  }, [location.search, location.state]);

  if (loading) {
    return (
      <>
        <Topbar title="Kết quả phân tích" subtitle="Đang tải dữ liệu..." />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-muted-foreground text-sm">Đang tải...</div>
        </main>
      </>
    );
  }

  if (!prediction) {
    return (
      <>
        <Topbar title="Kết quả phân tích" subtitle="Chưa có dữ liệu" />
        <main className="flex-1 flex items-center justify-center px-4">
          <div className="text-center space-y-4 max-w-sm">
            <div className="text-muted-foreground text-sm">
              Chưa có kết quả phân tích. Vui lòng upload file CSV trước.
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
      <Topbar title="Kết quả phân tích" subtitle="Dự đoán theo từng luồng từ batch suy luận mới nhất" />
      <main className="flex-1 w-full px-4 md:px-6 lg:px-8 py-4 md:py-6 space-y-4 lg:space-y-6">
        {prediction.isPreview && (
          <div className="rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning">
            {LARGE_RESULT_WARNING}
          </div>
        )}

        <SummaryCards summary={prediction.summary} />

        <div className="w-full min-w-0 overflow-hidden">
          <div className="w-full">
            <ErrorHistogram histogram={prediction.histogram} threshold={prediction.summary.threshold} />
          </div>
        </div>

        <div className="overflow-hidden">
          <AlertTable flows={prediction.alertFlows} />
        </div>

        <div className="overflow-hidden">
          <FlowTable
            flows={prediction.flows}
            isPreview={prediction.isPreview}
            totalRows={prediction.originalResultCount}
          />
        </div>
      </main>
    </>
  );
}
