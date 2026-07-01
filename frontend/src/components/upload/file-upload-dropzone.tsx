import { useRef, useState } from "react";
import { UploadCloud, FileText, X } from "lucide-react";

type Props = {
  onFile: (file: File) => void;
};

export function FileUploadDropzone({ onFile }: Props) {
  const [dragging, setDragging] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = (file: File) => {
    setSelected(file);
    onFile(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) accept(file);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) accept(file);
  };

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 cursor-pointer transition ${
        dragging ? "border-primary bg-primary/5" : "border-border bg-muted/20 hover:border-primary/50 hover:bg-muted/30"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={onInputChange}
      />
      {selected ? (
        <>
          <FileText className="h-8 w-8 text-primary" />
          <div className="text-sm font-medium">{selected.name}</div>
          <div className="text-xs text-muted-foreground">{(selected.size / 1024).toFixed(1)} KB</div>
          <button
            onClick={(e) => { e.stopPropagation(); setSelected(null); }}
            className="absolute top-3 right-3 h-6 w-6 grid place-items-center rounded-full border border-border bg-card hover:bg-muted"
          >
            <X className="h-3 w-3" />
          </button>
        </>
      ) : (
        <>
          <UploadCloud className="h-8 w-8 text-muted-foreground" />
          <div className="text-sm font-medium">Thả file CSV vào đây hoặc nhấp để chọn</div>
          <div className="text-xs text-muted-foreground">Chấp nhận file CSV định dạng luồng mạng CICIDS2017</div>
        </>
      )}
    </div>
  );
}
