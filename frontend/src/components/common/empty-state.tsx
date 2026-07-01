// Temporary placeholder component for pages under construction.

type EmptyStateProps = {
  title?: string;
  body?: string;
  description?: string;
};

export function EmptyState({
  title = "No data",
  body,
  description,
}: EmptyStateProps) {
  const text = body ?? description ?? "Nothing to display yet.";
  return (
    <div className="rounded-xl border border-border bg-card p-10 text-center">
      <div className="text-base font-semibold">{title}</div>
      <p className="mt-2 text-sm text-muted-foreground">{text}</p>
    </div>
  );
}

export default EmptyState;
