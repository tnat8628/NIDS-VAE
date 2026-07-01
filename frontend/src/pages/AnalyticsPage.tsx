import { Topbar } from "@/components/common/topbar";
import { EmptyState } from "@/components/common/empty-state";
import { BarChart3, Flame, Map, History } from "lucide-react";

export default function AnalyticsPage() {
  const cards = [
    { i: History, t: "Xu hướng lịch sử", b: "Tỷ lệ bất thường dài hạn, độ lệch phân bố, hiệu suất mô hình theo thời gian." },
    { i: Flame, t: "Bản đồ nhiệt tấn công", b: "Mật độ bất thường theo ASN nguồn, cổng đích và giao thức." },
    { i: Map, t: "Tương quan địa lý", b: "Địa lý hóa IP, đột biến bất thường theo vùng, phân tích tuyến đường." },
    { i: BarChart3, t: "So sánh mô hình", b: "Chỉ số song song giữa các thế hệ VAE và các lần huấn luyện lại." },
  ];
  return (
    <>
      <Topbar title="Phân tích" subtitle="Góc nhìn chuyên sâu và thông tin lịch sử" />
      <main className="flex-1 px-4 md:px-6 lg:px-8 py-4 md:py-6 space-y-4 lg:space-y-6 max-w-[1280px] mx-auto w-full">
        <EmptyState
          title="Không gian phân tích sắp ra mắt"
          body="Xu hướng bất thường lịch sử, lịch sử quét, phân tích tấn công, bản đồ nhiệt và so sánh mô hình sẽ hiển thị tại đây."
        />
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {cards.map((c) => {
            const Icon = c.i;
            return (
              <div key={c.t} className="relative rounded-xl border border-border bg-card p-5 overflow-hidden">
                <div className="absolute top-3 right-3 text-[9px] uppercase tracking-[0.14em] px-1.5 py-0.5 rounded-full bg-violet/15 text-violet border border-violet/30">Sắp có</div>
                <Icon className="h-5 w-5 text-primary" />
                <h3 className="mt-3 text-sm font-semibold">{c.t}</h3>
                <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{c.b}</p>
              </div>
            );
          })}
        </div>
      </main>
    </>
  );
}
