import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowUpRight, Eye, Loader2, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Topbar } from "@/components/common/topbar";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { deleteUpload, getDashboardOverview, getUploads } from "@/lib/api";
import type {
  DashboardOverviewResponse,
  UploadFilter,
  UploadListItem,
  UploadListResponse,
} from "@/types/api";

const PAGE_SIZE = 20;
const FILTERS: Array<{ value: UploadFilter; label: string }> = [
  { value: "all", label: "Tất cả" },
  { value: "analyzed", label: "Đã phân tích" },
  { value: "pending", label: "Chưa phân tích" },
];

function parseFilter(value: string | null): UploadFilter {
  return value === "analyzed" || value === "pending" ? value : "all";
}

function formatDate(value: string | null): string {
  if (!value) return "Chưa có";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Chưa có" : date.toLocaleString("vi-VN");
}

export default function UploadsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const filter = parseFilter(searchParams.get("filter"));
  const [page, setPage] = useState(1);
  const [uploads, setUploads] = useState<UploadListResponse | null>(null);
  const [overview, setOverview] = useState<DashboardOverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<UploadListItem | null>(null);
  const [deleting, setDeleting] = useState(false);

  const loadData = useCallback(
    async (targetPage = page) => {
      setLoading(true);
      setError(null);
      try {
        const [uploadData, overviewData] = await Promise.all([
          getUploads(targetPage, PAGE_SIZE, filter),
          getDashboardOverview(),
        ]);
        setUploads(uploadData);
        setOverview(overviewData);
      } catch {
        setError("Không thể tải danh sách file CSV từ PostgreSQL.");
      } finally {
        setLoading(false);
      }
    },
    [filter, page],
  );

  useEffect(() => {
    setPage(1);
  }, [filter]);

  useEffect(() => {
    loadData(page);
  }, [loadData, page]);

  const pagination = uploads?.pagination;
  const canGoBack = Boolean(pagination?.has_previous);
  const canGoNext = Boolean(pagination?.has_next);

  const overviewText = useMemo(() => {
    if (!overview) return "Overview sẽ được đồng bộ sau mỗi thao tác xóa.";
    return `${overview.uploads.total_uploads.toLocaleString()} file · ${overview.uploads.total_uploaded_flows.toLocaleString()} flow upload · ${overview.analysis.total_analyzed_flows.toLocaleString()} flow đã phân tích`;
  }, [overview]);

  async function confirmDelete() {
    if (!deleteTarget || deleting) return;
    setDeleting(true);
    try {
      await deleteUpload(deleteTarget.upload_id);
      toast.success(`Đã xóa file ${deleteTarget.filename} và dữ liệu liên quan.`);
      const currentItems = uploads?.items.length ?? 0;
      const nextPage = currentItems <= 1 && page > 1 ? page - 1 : page;
      setDeleteTarget(null);
      setPage(nextPage);
      await loadData(nextPage);
    } catch {
      toast.error(`Không thể xóa file ${deleteTarget.filename}. Vui lòng thử lại.`);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <>
      <Topbar
        title="Quản lý file CSV"
        subtitle="Danh sách file đã upload từ PostgreSQL, kèm trạng thái phân tích và thao tác xóa an toàn"
      />
      <main className="flex-1 w-full px-4 md:px-6 lg:px-8 py-4 md:py-6 space-y-4 lg:space-y-6">
        <div className="rounded-xl border border-border bg-card p-4 shadow-soft">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-sm font-semibold">Dữ liệu upload</h2>
              <p className="mt-1 text-xs text-muted-foreground">{overviewText}</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {FILTERS.map((item) => (
                <Button
                  key={item.value}
                  type="button"
                  size="sm"
                  variant={filter === item.value ? "default" : "outline"}
                  onClick={() => {
                    setSearchParams(item.value === "all" ? {} : { filter: item.value });
                  }}
                >
                  {item.label}
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
              Đang tải danh sách file...
            </div>
          ) : error ? (
            <div className="p-6 text-sm text-destructive">{error}</div>
          ) : uploads && uploads.items.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Tên file</TableHead>
                  <TableHead>Thời điểm upload</TableHead>
                  <TableHead className="text-right">Số flow</TableHead>
                  <TableHead className="text-right">Số cột</TableHead>
                  <TableHead>Trạng thái</TableHead>
                  <TableHead>Lần phân tích gần nhất</TableHead>
                  <TableHead className="text-right">Anomaly latest run</TableHead>
                  <TableHead className="text-right">Hành động</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {uploads.items.map((item) => {
                  const resultsUrl =
                    item.latest_run_id
                      ? `/results?uploadId=${encodeURIComponent(item.upload_id)}&runId=${encodeURIComponent(item.latest_run_id)}`
                      : null;
                  return (
                    <TableRow key={item.upload_id}>
                      <TableCell className="max-w-[260px] truncate font-medium">
                        {item.filename}
                      </TableCell>
                      <TableCell className="whitespace-nowrap text-muted-foreground">
                        {formatDate(item.created_at)}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {item.row_count.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {item.col_count.toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <span
                          className={`inline-flex rounded-full px-2 py-1 text-xs ${
                            item.analysis_status === "completed"
                              ? "bg-success/10 text-success"
                              : "bg-muted text-muted-foreground"
                          }`}
                        >
                          {item.analysis_status === "completed" ? "Đã phân tích" : "Chưa phân tích"}
                        </span>
                      </TableCell>
                      <TableCell className="whitespace-nowrap text-muted-foreground">
                        {formatDate(item.latest_predicted_at)}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {item.analysis_status === "completed" ? item.anomaly_count.toLocaleString() : "—"}
                      </TableCell>
                      <TableCell>
                        <div className="flex justify-end gap-2">
                          {resultsUrl && (
                            <Button asChild size="sm" variant="outline">
                              <Link to={resultsUrl}>
                                <Eye className="h-4 w-4" />
                                Xem kết quả
                              </Link>
                            </Button>
                          )}
                          <Button
                            type="button"
                            size="icon"
                            variant="ghost"
                            className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                            title="Xóa file"
                            aria-label={`Xóa file ${item.filename}`}
                            onClick={() => setDeleteTarget(item)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          ) : (
            <div className="flex flex-col items-center justify-center gap-3 py-14 text-center">
              <p className="text-sm text-muted-foreground">Chưa có file CSV phù hợp bộ lọc hiện tại.</p>
              <Button asChild variant="outline" size="sm">
                <Link to="/upload">
                  Tải lên CSV mới <ArrowUpRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          )}
        </div>

        {pagination && pagination.total_items > 0 && (
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="text-xs text-muted-foreground">
              Trang {pagination.page} / {Math.max(1, pagination.total_pages)} ·{" "}
              {pagination.total_items.toLocaleString()} file
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={!canGoBack || loading}
                onClick={() => setPage((value) => Math.max(1, value - 1))}
              >
                Trước
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={!canGoNext || loading}
                onClick={() => setPage((value) => value + 1)}
              >
                Sau
              </Button>
            </div>
          </div>
        )}
      </main>

      <AlertDialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => {
          if (!open && !deleting) setDeleteTarget(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Xóa file đã tải lên?</AlertDialogTitle>
            <AlertDialogDescription className="space-y-2">
              <span className="block">
                Bạn sắp xóa file {deleteTarget?.filename} gồm{" "}
                {deleteTarget?.row_count.toLocaleString()} flow.
              </span>
              <span className="block">
                Toàn bộ dữ liệu gốc, lịch sử phân tích và kết quả dự đoán liên quan sẽ bị xóa vĩnh viễn.
              </span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Hủy</AlertDialogCancel>
            <Button
              type="button"
              variant="destructive"
              disabled={deleting}
              onClick={confirmDelete}
            >
              {deleting && <Loader2 className="h-4 w-4 animate-spin" />}
              Xóa vĩnh viễn
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}