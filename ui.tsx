/**
 * React components for the artifacts `artifacts.py` pushes out of the sandbox.
 *
 * Three components, routed by file extension on the Python side:
 *
 *   chart  images (.png/.jpg/.svg/...) - matplotlib output, rendered inline
 *   table  tabular (.csv/.tsv/.xlsx)   - parsed and previewed, with full download
 *   file   everything else             - download card
 *
 * Every component receives the file's bytes as base64 in `props.data` and needs no
 * network of its own. Styles are inline rather than Tailwind classes so these render
 * identically in agent-chat-ui, LangGraph Studio, or any other frontend that mounts
 * the bundle — none of which are guaranteed to share a stylesheet with us.
 */

import React, { useMemo, useState } from "react";
import * as XLSX from "xlsx";

type ArtifactProps = {
  name: string;
  path: string;
  size: number;
  mime: string;
  suffix: string;
  data: string | null;
  too_large: boolean;
};

// How many rows a table preview draws before it stops. The full file is always in the
// download; this only bounds what we paint. A 200-paper corpus exported to xlsx is a
// realistic output here and 200 DOM tables' worth of rows would jank the transcript.
const PREVIEW_ROWS = 100;

const styles: Record<string, React.CSSProperties> = {
  card: {
    borderRadius: "8px",
    border: "1px solid #e5e7eb",
    backgroundColor: "#ffffff",
    padding: "16px",
    marginTop: "8px",
    maxWidth: "100%",
    fontFamily:
      "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
    marginBottom: "12px",
  },
  nameRow: { display: "flex", alignItems: "center", gap: "8px", minWidth: 0 },
  name: {
    fontSize: "14px",
    fontWeight: 600,
    color: "#111827",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  meta: { fontSize: "12px", color: "#6b7280", whiteSpace: "nowrap" },
  button: {
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    borderRadius: "6px",
    backgroundColor: "#2563eb",
    color: "#ffffff",
    padding: "6px 12px",
    fontSize: "13px",
    fontWeight: 500,
    border: "none",
    cursor: "pointer",
    textDecoration: "none",
    whiteSpace: "nowrap",
  },
  image: {
    maxWidth: "100%",
    height: "auto",
    borderRadius: "6px",
    display: "block",
  },
  tableWrap: {
    overflowX: "auto",
    border: "1px solid #e5e7eb",
    borderRadius: "6px",
    maxHeight: "420px",
    overflowY: "auto",
  },
  table: { borderCollapse: "collapse", fontSize: "12px", width: "100%" },
  th: {
    padding: "6px 10px",
    textAlign: "left",
    fontWeight: 600,
    color: "#374151",
    backgroundColor: "#f9fafb",
    borderBottom: "1px solid #e5e7eb",
    position: "sticky",
    top: 0,
    whiteSpace: "nowrap",
  },
  td: {
    padding: "6px 10px",
    color: "#1f2937",
    borderBottom: "1px solid #f3f4f6",
    verticalAlign: "top",
  },
  note: { fontSize: "12px", color: "#6b7280", marginTop: "8px" },
  error: { fontSize: "13px", color: "#b91c1c" },
};

function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/**
 * Download without a network round trip — the bytes are already here, so a blob URL
 * is the whole mechanism. `download` carries the original filename through.
 */
function DownloadButton({ name, mime, data }: Pick<ArtifactProps, "name" | "mime" | "data">) {
  const href = useMemo(() => {
    if (!data) return null;
    return URL.createObjectURL(new Blob([base64ToBytes(data)], { type: mime }));
  }, [data, mime]);

  if (!href) return null;
  return (
    <a style={styles.button} href={href} download={name}>
      Download
    </a>
  );
}

function Shell({
  artifact,
  children,
}: {
  artifact: ArtifactProps;
  children?: React.ReactNode;
}) {
  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={styles.nameRow}>
          <span style={styles.name} title={artifact.path}>
            {artifact.name}
          </span>
          <span style={styles.meta}>{formatBytes(artifact.size)}</span>
        </div>
        <DownloadButton {...artifact} />
      </div>
      {artifact.too_large ? (
        <p style={styles.note}>
          Too large to preview inline. It is still in the sandbox at{" "}
          <code>{artifact.path}</code> until the session ends.
        </p>
      ) : (
        children
      )}
    </div>
  );
}

/** Images: matplotlib PNGs, JPEG charts, SVG. */
const Chart = (artifact: ArtifactProps) => {
  const [failed, setFailed] = useState(false);
  return (
    <Shell artifact={artifact}>
      {artifact.data && !failed ? (
        <img
          style={styles.image}
          src={`data:${artifact.mime};base64,${artifact.data}`}
          alt={artifact.name}
          onError={() => setFailed(true)}
        />
      ) : (
        <p style={styles.error}>Could not render this image — use Download instead.</p>
      )}
    </Shell>
  );
};

/**
 * Minimal RFC 4180 parser: handles quoted fields, embedded delimiters, embedded
 * newlines and doubled quotes.
 *
 * A `split(",")` would be shorter and would corrupt this corpus specifically —
 * paper titles contain commas constantly ("CRISPR-Cas9, base editors, and prime
 * editing") and every such row would shift a column.
 */
function parseDelimited(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];

    if (quoted) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      quoted = true;
    } else if (char === delimiter) {
      row.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      // Swallow the \n of a \r\n pair rather than emitting a phantom empty row.
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.length > 1 || r[0] !== "");
}

function toRows(artifact: ArtifactProps): string[][] {
  if (!artifact.data) return [];
  const bytes = base64ToBytes(artifact.data);

  if (artifact.suffix === ".csv" || artifact.suffix === ".tsv") {
    const text = new TextDecoder().decode(bytes);
    return parseDelimited(text, artifact.suffix === ".tsv" ? "\t" : ",");
  }

  // xlsx/xls. `header: 1` gives raw row arrays so the header row is just row 0 and we
  // render whatever the sheet actually has, rather than guessing at column names.
  const workbook = XLSX.read(bytes, { type: "array" });
  const sheet = workbook.Sheets[workbook.SheetNames[0]];
  if (!sheet) return [];
  const rows = XLSX.utils.sheet_to_json<unknown[]>(sheet, {
    header: 1,
    blankrows: false,
    defval: "",
  });
  return rows.map((r) => r.map((cell) => (cell == null ? "" : String(cell))));
}

/** Tabular files: CSV/TSV parsed inline, Excel parsed with SheetJS. */
const Table = (artifact: ArtifactProps) => {
  const { rows, error } = useMemo(() => {
    try {
      return { rows: toRows(artifact), error: null as string | null };
    } catch (e) {
      return { rows: [] as string[][], error: (e as Error).message };
    }
  }, [artifact.data, artifact.suffix]);

  if (error) {
    return (
      <Shell artifact={artifact}>
        <p style={styles.error}>Could not parse this file ({error}).</p>
      </Shell>
    );
  }
  if (rows.length === 0) {
    return (
      <Shell artifact={artifact}>
        <p style={styles.note}>Empty file.</p>
      </Shell>
    );
  }

  const [header, ...body] = rows;
  const shown = body.slice(0, PREVIEW_ROWS);

  return (
    <Shell artifact={artifact}>
      <div style={styles.tableWrap}>
        <table style={styles.table}>
          <thead>
            <tr>
              {header.map((cell, i) => (
                <th key={i} style={styles.th}>
                  {cell}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r, ri) => (
              <tr key={ri}>
                {header.map((_, ci) => (
                  <td key={ci} style={styles.td}>
                    {r[ci] ?? ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={styles.note}>
        {body.length > shown.length
          ? `Showing ${shown.length} of ${body.length} rows — download for the full file.`
          : `${body.length} row${body.length === 1 ? "" : "s"}.`}
      </p>
    </Shell>
  );
};

/** Anything without a preview: JSON, PDF, .txt, .md. */
const File = (artifact: ArtifactProps) => <Shell artifact={artifact} />;

export default {
  chart: Chart,
  table: Table,
  file: File,
};
