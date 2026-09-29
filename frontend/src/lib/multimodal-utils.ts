import { ContentBlock } from "@langchain/core/messages";
import { toast } from "sonner";

// setup: the attachments this agent can do something with. None of them stay in model
// context — each rides in as a file block carrying its filename, and the graph takes the
// payload back out and materialises it in the sandbox (deep_life_sci/middleware/uploads.py,
// whose UPLOAD_KINDS is the server-side half of this list).
//
// Extension first and MIME second, deliberately — see the note in use-file-upload.tsx.
// `.xls` is in neither list: reading it needs xlrd, which is not in the sandbox snapshot,
// and the sandbox blocks runtime installs, so it is refused at the composer rather than
// failing deep inside a run. `application/vnd.ms-excel` is left out for the same reason,
// even though a Windows .csv arrives claiming it — the extension check has already passed
// that one by the time MIME is consulted.
export const UPLOAD_SUFFIXES = [
  // tables, gzipped or not
  ".csv", ".tsv", ".txt", ".xlsx", ".xlsm",
  ".csv.gz", ".tsv.gz", ".txt.gz",
  // bibliographies
  ".nbib", ".medline", ".ris", ".bib", ".bibtex",
  // a paper the agent cannot fetch for itself
  ".pdf",
  // compound sets
  ".sdf", ".sdf.gz", ".mol", ".smi", ".smiles",
  // sequences
  ".fasta", ".fa", ".fna", ".faa", ".fasta.gz", ".gb", ".gbk", ".genbank",
  // figures, gels, panels
  ".png", ".jpg", ".jpeg", ".gif", ".webp",
];

// MIME is the fallback, so this only needs the types a browser reliably reports for the
// list above. Anything it gets wrong is caught by the extension check first.
export const UPLOAD_TYPES = [
  "text/csv",
  "text/plain",
  "text/tab-separated-values",
  "application/gzip",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.ms-excel.sheet.macroEnabled.12",
  "application/pdf",
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
];

export function isSandboxUpload(file: File): boolean {
  const name = file.name.toLowerCase();
  if (UPLOAD_SUFFIXES.some((suffix) => name.endsWith(suffix))) return true;
  return UPLOAD_TYPES.includes(file.type);
}

// Normalised off the extension, because the browser's value is the unreliable half and the
// server keys on the extension as well. A type the browser reported is better than nothing
// for anything not listed here — the graph never reads it, but the dedupe check does.
export function uploadMimeType(file: File): string {
  const name = file.name.toLowerCase();
  if (name.endsWith(".xlsx"))
    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
  if (name.endsWith(".xlsm")) return "application/vnd.ms-excel.sheet.macroEnabled.12";
  if (name.endsWith(".pdf")) return "application/pdf";
  if (name.endsWith(".gz")) return "application/gzip";
  if (name.endsWith(".tsv")) return "text/tab-separated-values";
  if (name.endsWith(".csv")) return "text/csv";
  if (name.endsWith(".png")) return "image/png";
  if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
  if (name.endsWith(".gif")) return "image/gif";
  if (name.endsWith(".webp")) return "image/webp";
  return file.type || "text/plain";
}

// Returns a Promise of a typed multimodal block for images or PDFs
export async function fileToContentBlock(
  file: File,
): Promise<ContentBlock.Multimodal.Data> {
  const supportedImageTypes = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
  ];
  const supportedFileTypes = [...supportedImageTypes, "application/pdf"];

  if (isSandboxUpload(file)) {
    return {
      type: "file",
      mimeType: uploadMimeType(file),
      data: await fileToBase64(file),
      metadata: { filename: file.name },
    };
  }

  if (!supportedFileTypes.includes(file.type)) {
    toast.error(
      `Unsupported file type: ${file.type}. Supported types are: ${supportedFileTypes.join(", ")}`,
    );
    return Promise.reject(new Error(`Unsupported file type: ${file.type}`));
  }

  const data = await fileToBase64(file);

  if (supportedImageTypes.includes(file.type)) {
    return {
      type: "image",
      mimeType: file.type,
      data,
      metadata: { name: file.name },
    };
  }

  // PDF
  return {
    type: "file",
    mimeType: "application/pdf",
    data,
    metadata: { filename: file.name },
  };
}

// Helper to convert File to base64 string
export async function fileToBase64(file: File): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Remove the data:...;base64, prefix
      resolve(result.split(",")[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Type guard for Base64ContentBlock
export function isBase64ContentBlock(
  block: unknown,
): block is ContentBlock.Multimodal.Data {
  if (typeof block !== "object" || block === null || !("type" in block))
    return false;
  // any accepted upload — transport for the graph rather than model context
  if (
    (block as { type: unknown }).type === "file" &&
    "mimeType" in block &&
    typeof (block as { mimeType?: unknown }).mimeType === "string" &&
    UPLOAD_TYPES.includes((block as { mimeType: string }).mimeType)
  ) {
    return true;
  }
  // file type (legacy)
  if (
    (block as { type: unknown }).type === "file" &&
    "mimeType" in block &&
    typeof (block as { mimeType?: unknown }).mimeType === "string" &&
    ((block as { mimeType: string }).mimeType.startsWith("image/") ||
      (block as { mimeType: string }).mimeType === "application/pdf")
  ) {
    return true;
  }
  // image type (new)
  if (
    (block as { type: unknown }).type === "image" &&
    "mimeType" in block &&
    typeof (block as { mimeType?: unknown }).mimeType === "string" &&
    (block as { mimeType: string }).mimeType.startsWith("image/")
  ) {
    return true;
  }
  return false;
}
