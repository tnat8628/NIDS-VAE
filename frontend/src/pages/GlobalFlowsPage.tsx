import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowUpRight, Loader2, RefreshCw } from "lucide-react";
import { Topbar } from "@/components/common/topbar";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getDashboardFlows, getDashboardOverview } from "@/lib/api";
import type {
  DashboardOverviewResponse,
  GlobalFlowListResponse,
  GlobalPredictionFilter,
} from "@/types/api";

const PAGE_SIZE = 25;
const FILTERS: Array<{ value: GlobalPredictionFilter; label: string; to: string }> = [
  { value: "all", label: "All", to: "/flows" },
  { value: "anomaly", label: "Anomaly", to: "/flows?prediction=anomaly" },
  { value: "normal", label: "Normal", to: "/flows?prediction=normal" },
];

function parsePrediction(value: string | null): GlobalPredictionFilter {
  return value === "anomaly" || value === "normal" ? value : "all";
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Chưa có" : date.toLocaleString("vi-VN");
}

export default function GlobalFlowsPage() {
  const [searchParams] = useSearchParams();
  const prediction = parsePrediction(searchParams.get("prediction"));
  const [page, setPage] = useState(1);
  const [flows, setFlows] = useState<GlobalFlowListResponse | null>(null);
  const [overview, setOverview] = useState<DashboardOverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(
    async (targetPage = page) => {
      setLoading(true);
      setError(null);
      try {
        const [flowData, overviewData] = await Promise.all([
          getDashboardFlows(targetPage, PAGE_SIZE, prediction),
          getDashboardOverview(),
        ]);
        setFlows(flowData);
        setOverview(overviewData);
      } catch {
        setError("Không thể tải danh sách flow toàn hệ thống từ PostgreSQL.");
      } finally {
        setLoading(false);
      }
    },
    [page, prediction],
  );

  useEffect(() => {
    setPage(1);
  }, [prediction]);

  useEffect(() => {
    loadData(page);
  }, [loadData, page]);

  const summaryText = useMemo(() => {
    if (!overview) return "Chỉ hiển thị latest successful inference run của mỗi file.";
    return `${overview.analysis.anomaly_count.toLocaleString()} anomaly · ${overview.analysis.normal_count.toLocaleString()} normal · ${(overview.analysis.anomaly_rate * 100).toFixed(2)}% anomaly rate`;
  }, [overview]);

  const pagination = flows?.pagination;

  return (
    <>
      <Topbar
        title="Luồng toàn hệ thống"
        subtitle="Flow explorer database-backed, không double count khi một file được predict lại"
      />
      <main className="flex-1 w-full px-4 md:px-6 lg:px-8 py-4 md:py-6 space-y-4 lg:space-y-6">
        <div className="rounded-xl border border-border bg-card p-4 shadow-soft">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-sm font-semibold">Anomaly rate toàn hệ thống</h2>
              <p className="mt-1 text-xs text-muted-foreground">{summaryText}</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {FILTERS.map((item) => (
                <Button
                  key={item.value}
                  asChild
                  size="sm"
                  variant={prediction === item.value ? "default" : "outline"}
                >
                  <Link to={item.to}>{item.label}</Link>
                </Button>
              ))}
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => loadData(page)}
                disabled={loading}
              >
                <RefreshCw className={loading ? "animate-spin" : ""} />
                Làm mới
              </Button>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card shadow-soft overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center gap-2 py-14 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Đang tải flow...
            </div>
          ) : error ? (
            <div className="p-6 text-sm text-destructive">{error}</div>
          ) : flows && flows.items.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>File</TableHead>
                  <TableHead>Run ID</TableHead>
                  <TableHead className="text-right">Row index</TableHead>
                  <TableHead className="text-right">Reconstruction error</TableHead>
                  <TableHead>Prediction</TableHead>
                  <TableHead>Thời điểm</TableHead>
                  <TableHead className="text-right">Kết quả file</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {flows.items.map((item) => {
                  const resultsUrl = `/results?uploadId=${encodeURIComponent(item.upload_id)}&runId=${encodeURIComponent(item.run_id)}`;
                  return (
                    <TableRow key={`${item.run_id}-${item.row_index}`}>
                      <TableCell className="max-w-[260px] truncate font-medium">
                        {item.filename}
                      </TableCell>
                      <TableCell className="max-w-[180px] truncate font-mono text-xs text-muted-foreground">
                        {item.run_id}
                      </TableCell>
                      <TableCell className="text-right font-mono">{item.row_index}</TableCell>
                      <TableCell className="text-right font-mono">
                        {item.reconstruction_error.toFixed(6)}
                      </TableCell>
                      <TableCell>
                        <span
                          className={`inline-flex rounded-full px-2 py-1 text-xs ${
                            item.prediction === 1
                              ? "bg-anomaly/10 text-anomaly"
                              : "bg-success/10 text-success"
                          }`}
                        >
                          {item.prediction_label}
                        </span>
                      </TableCell>
                      <TableCell className="whitespace-nowrap text-muted-foreground">
                        {formatDate(item.created_at)}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button asChild variant="outline" size="sm">
                          <Link to={resultsUrl}>
                            Mở file <ArrowUpRight className="h-4 w-4" />
                          </Link>
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          ) : (
            <div className="flex items-center justify-center py-14 text-sm text-muted-foreground">
              Chưa có flow phù hợp bộ lọc hiện tại.
            </div>
          )}
        </div>

        {pagination && pagination.total_items > 0 && (
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="text-xs text-muted-foreground">
              Trang {pagination.page} / {Math.max(1, pagination.total_pages)} ·{" "}
              {pagination.total_items.toLocaleString()} flow
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={!pagination.has_previous || loading}
                onClick={() => setPage((value) => Math.max(1, value - 1))}
              >
                Trước
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={!pagination.has_next || loading}
                onClick={() => setPage((value) => value + 1)}
              >
                Sau
              </Button>
            </div>
          </div>
        )}
      </main>
    </>
  );
}