import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Upload } from "lucide-react";
import { Topbar } from "@/components/common/topbar";
import { SummaryCards } from "@/components/dashboard/summary-cards";
import { ErrorHistogram } from "@/components/dashboard/error-histogram";
import { FlowTable } from "@/components/dashboard/flow-table";
import { AlertTable } from "@/components/dashboard/alert-table";
import { getUploadResults } from "@/lib/api";
import {
  mapPaginatedResults,
  type MappedPrediction,
} from "@/lib/mapper";
import type { PaginatedPredictionResponse } from "@/types/api";

const PAGE_SIZE = 25;

export default function ResultsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const uploadId = searchParams.get("uploadId");
  const inferenceRunId = searchParams.get("runId");
  const requestedPage = Number(searchParams.get("page") ?? "1");
  const page = Number.isInteger(requestedPage) && requestedPage > 0 ? requestedPage : 1;

  const [response, setResponse] = useState<PaginatedPredictionResponse | null>(null);
  const [loading, setLoading] = useState(Boolean(uploadId && inferenceRunId));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!uploadId || !inferenceRunId) {
      setLoading(false);
      setResponse(null);
      setError("URL kết quả phải có đầy đủ uploadId và runId.");
      return;
    }

    let active = true;
    setLoading(true);
    setError(null);

    getUploadResults(uploadId, page, PAGE_SIZE, inferenceRunId)
      .then((data) => {
        if (active) setResponse(data);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        const message =
          (requestError as { response?: { data?: { message?: string; detail?: string } } })
            ?.response?.data?.message ??
          (requestError as { response?: { data?: { detail?: string } } })
            ?.response?.data?.detail ??
          "Không thể tải kết quả từ database.";
        setError(message);
        setResponse(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [inferenceRunId, page, uploadId]);

  const prediction = useMemo<MappedPrediction | null>(
    () => (response ? mapPaginatedResults(response) : null),
    [response],
  );

  function changePage(nextPage: number) {
    if (!uploadId || !inferenceRunId) return;
    const next = new URLSearchParams(searchParams);
    next.set("uploadId", uploadId);
    next.set("page", String(nextPage));
    next.set("runId", inferenceRunId);
    setSearchParams(next);
  }

  if (loading && !prediction) {
    return (
      <>
        <Topbar title="Kết quả phân tích" subtitle="Đang tải dữ liệu từ PostgreSQL..." />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-muted-foreground text-sm">Đang tải...</div>
        </main>
      </>
    );
  }

  if (!prediction || !response) {
    return (
      <>
        <Topbar title="Kết quả phân tích" subtitle="Chưa có dữ liệu" />
        <main className="flex-1 flex items-center justify-center px-4">
          <div className="text-center space-y-4 max-w-sm">
            <div className="text-muted-foreground text-sm">
              {error ?? "Chưa có kết quả phân tích. Vui lòng upload file CSV trước."}
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
      <Topbar
        title="Kết quả phân tích"
        subtitle="Kết quả phân tích của file hiện tại"
      />
      <main className="flex-1 w-full px-4 md:px-6 lg:px-8 py-4 md:py-6 space-y-4 lg:space-y-6">
        <SummaryCards summary={prediction.summary} />

        <div className="w-full min-w-0 overflow-hidden">
          <ErrorHistogram
            histogram={prediction.histogram}
            threshold={prediction.summary.threshold}
          />
        </div>

        <div className="overflow-hidden">
          <AlertTable flows={prediction.alertFlows} />
        </div>

        <div className="overflow-hidden">
          <FlowTable
            uploadId={uploadId!}
            inferenceRunId={inferenceRunId!}
            flows={prediction.flows}
            pagination={response.pagination}
            loading={loading}
            onPageChange={changePage}
          />
        </div>
      </main>
    </>
  );
}