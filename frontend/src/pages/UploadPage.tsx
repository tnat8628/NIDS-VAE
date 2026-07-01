import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { CheckCircle2, FileText, Loader2, Sparkles, Wand2 } from "lucide-react";
import { Topbar } from "@/components/common/topbar";
import { FileUploadDropzone } from "@/components/upload/file-upload-dropzone";
import { predictCsv } from "@/lib/api";
import { buildDashboardViewModel, savePredictionToStorage } from "@/lib/mapper";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  // Trang thai loading khi dang goi API /predict.
  const [loading, setLoading] = useState(false);
  // Loi hien thi truc tiep neu backend hoac ket noi that bai.
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  /**
   * Gui file CSV len POST /predict, tao view model nhe, roi chuyen den trang ket qua.
   * Khong day raw response.results vao route state vi file lon co the lam Chrome het bo nho.
   */
  async function handleAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const response = await predictCsv(file);
      // Tao view model nhe ngay lap tuc: summary + histogram + top alerts + preview rows.
      const predictionView = buildDashboardViewModel(response);

      // localStorage chi nhan cache da cat gioi han, khong stringify toan bo results.
      savePredictionToStorage(response);
      toast.success("Phân tích hoàn thành", {
        description: `Phát hiện ${response.summary?.anomaly_count ?? 0} bất thường trong ${response.summary?.total_flows ?? 0} luồng`,
      });

      // Navigation state cung chi giu ban gon nhe de ResultsPage khong giu payload lon trong RAM.
      navigate("/results", { state: { predictionView } });
    } catch (err: unknown) {
      const message =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        (err instanceof Error ? err.message : "Không thể kết nối backend. Vui lòng thử lại.");
      setError(message);
      toast.error("Phân tích thất bại", { description: message });
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Topbar title="Tải lên & Phân tích" subtitle="Nộp file CSV luồng mạng để sử dụng VAE" />
      <main className="flex-1 px-4 md:px-6 lg:px-8 py-4 md:py-6 max-w-[1280px] mx-auto w-full">
        <div className="max-w-4xl mx-auto grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <FileUploadDropzone onFile={(nextFile) => { setFile(nextFile); setError(null); }} />

            {error && (
              <div className="rounded-xl border border-anomaly/40 bg-anomaly/10 px-4 py-3 text-sm text-anomaly">
                {error}
              </div>
            )}

            <button
              disabled={!file || loading}
              onClick={handleAnalyze}
              className="w-full h-12 rounded-xl bg-gradient-primary text-primary-foreground font-medium shadow-glow disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Đang phân tích...
                </>
              ) : (
                <>
                  <Wand2 className="h-4 w-4" />
                  Chạy phát hiện bất thường VAE
                </>
              )}
            </button>
          </div>

          <aside className="space-y-4">
            <div className="rounded-xl border border-border bg-card p-5">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-semibold">Quy trình</h3>
              </div>
              <ol className="space-y-3 text-xs">
                {["Kiểm tra định dạng và schema CSV", "Tiền xử lý về đúng 66 đặc trưng", "Chuẩn hóa bằng StandardScaler đã huấn luyện", "Chạy suy luận qua mô hình VAE", "Tính reconstruction error", "So sánh với ngưỡng để phân loại Normal/Anomaly"].map((step, index) => (
                  <li key={step} className="flex items-center gap-2.5">
                    <span className="h-5 w-5 grid place-items-center rounded-full border border-border bg-background text-[10px] font-mono">{index + 1}</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            </div>

            <div className="rounded-xl border border-border bg-card p-5">
              <div className="flex items-center gap-2 mb-3">
                <FileText className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-semibold">Yêu cầu dữ liệu đầu vào</h3>
              </div>
              <ul className="space-y-1.5 text-xs text-muted-foreground font-mono">
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-success" /> File CSV định dạng flow-level</li>
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-success" /> Tương thích CICIDS2017</li>
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-success" />  Backend tự động kiểm tra schema</li>
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-success" /> Backend tự động chuẩn hóa về 66 đặc trưng VAE</li>
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-success" /> Tự động xử lý giá trị thiếu (NaN/Inf)</li>


              </ul>
            </div>
          </aside>
        </div>
      </main>
    </>
  );
}
