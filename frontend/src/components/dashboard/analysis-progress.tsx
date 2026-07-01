import { useEffect, useState } from "react";
import { Brain } from "lucide-react";

const stages = [
  "Đang phân tích CSV và kiểm tra schema...",
  "Đang chuẩn hóa đặc trưng bằng scaler đã huấn luyện...",
  "Đang chạy lượt tiến VAE...",
  "Đang tính lỗi tái tạo...",
  "Đang đánh giá ngưỡng phát hiện bất thường...",
  "Đang tạo thông tin bất thường...",
];

export function AnalysisProgress({ onDone }: { onDone?: () => void }) {
  const [step, setStep] = useState(0);
  const [pct, setPct] = useState(0);

  useEffect(() => {
    const t = setInterval(() => {
      setPct((p) => {
        const n = p + 2;
        const s = Math.min(stages.length - 1, Math.floor((n / 100) * stages.length));
        setStep(s);
        if (n >= 100) {
          clearInterval(t);
          onDone?.();
        }
        return Math.min(100, n);
      });
    }, 80);
    return () => clearInterval(t);
  }, [onDone]);

  return (
    <div className="fixed inset-0 z-50 bg-background/85 backdrop-blur-md grid place-items-center p-6">
      <div className="w-full max-w-md rounded-2xl border border-border bg-card p-8 shadow-glow relative overflow-hidden">
        <div className="absolute inset-0 bg-glow-field opacity-50 pointer-events-none" />
        <div className="relative">
          <div className="mx-auto h-16 w-16 rounded-2xl bg-gradient-primary grid place-items-center shadow-glow animate-float">
            <Brain className="h-8 w-8 text-primary-foreground" />
          </div>
          <h3 className="mt-5 text-center text-base font-semibold tracking-tight">AI đang suy luận</h3>
          <p className="mt-1 text-center text-xs text-muted-foreground">{stages[step]}</p>

          <div className="mt-6 h-1.5 rounded-full bg-muted overflow-hidden">
            <div className="h-full bg-gradient-primary transition-all" style={{ width: `${pct}%` }} />
          </div>
          <div className="mt-2 flex justify-between text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
            <span>Suy luận VAE</span>
            <span className="font-mono">{pct}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
