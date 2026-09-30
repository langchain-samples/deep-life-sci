import { useState, useRef, useEffect, ChangeEvent } from "react";
import { toast } from "sonner";
import { ContentBlock } from "@langchain/core/messages";
import {
  fileToContentBlock,
  isSandboxUpload,
  uploadMimeType,
  UPLOAD_TYPES,
} from "@/lib/multimodal-utils";

// Everything deep_life_sci/middleware/uploads.py has a reader for. None of it is
// model context — the graph lifts the payload off the human message before the first model
// call and materialises it in the sandbox at /workspace/uploads, so a table gets computed
// over, a bibliography becomes a corpus, and a PDF or an image is read by a cheap subagent
// instead of by the root model. See frontend/CLAUDE.md.
export const SUPPORTED_FILE_TYPES: string[] = [...UPLOAD_TYPES];

// Every call site below tests these rather than the list, because a MIME-only check rejects
// the file the user came to attach: Windows with Excel installed reports a .csv as
// `application/vnd.ms-excel`, and some browsers report "" or application/octet-stream.
export function isSupportedUpload(file: File): boolean {
  return isSandboxUpload(file);
}

// Which uploads become a `type: "file"` block rather than an image one: all of them. An
// image is transport to the sandbox here rather than model context, so it takes the same
// shape as everything else and the graph has one block type to strip.
export function isFileBlockUpload(file: File): boolean {
  return isSandboxUpload(file);
}

export const UNSUPPORTED_FILE_TITLE = "That file type isn't supported";
export const UNSUPPORTED_FILE_BODY =
  "Tables, bibliographies (.nbib/.ris/.bib), PDFs, .sdf/.smi, FASTA/GenBank and images.";

interface UseFileUploadOptions {
  initialBlocks?: ContentBlock.Multimodal.Data[];
}

// A deployed agent refuses any request over 25 MB, and attachments travel inside the run
// request as base64, a third larger than the files. Kept under that with room for the rest
// of the message; a local server has no limit, but one rule is less surprising than two.
const MAX_ATTACHMENT_BYTES = 24 * 1024 * 1024;

function attachedBytes(blocks: ContentBlock.Multimodal.Data[]): number {
  return blocks.reduce(
    (total, b) => total + (typeof b.data === "string" ? b.data.length : 0),
    0,
  );
}

export function useFileUpload({
  initialBlocks = [],
}: UseFileUploadOptions = {}) {
  const [contentBlocks, setContentBlocks] =
    useState<ContentBlock.Multimodal.Data[]>(initialBlocks);
  const dropRef = useRef<HTMLDivElement>(null);

  // Checked against the blocks this render has, like the duplicate check below, so the
  // toast is not fired from inside a state updater.
  const addBlocks = (newBlocks: ContentBlock.Multimodal.Data[]) => {
    if (attachedBytes([...contentBlocks, ...newBlocks]) > MAX_ATTACHMENT_BYTES) {
      toast.error("Attachments too large", {
        description:
          "One message can carry about 18 MB of files in total. Send the rest in a follow-up message.",
      });
      return;
    }
    setContentBlocks((prev) => [...prev, ...newBlocks]);
  };
  const [dragOver, setDragOver] = useState(false);
  const dragCounter = useRef(0);

  const isDuplicate = (file: File, blocks: ContentBlock.Multimodal.Data[]) => {
    if (isFileBlockUpload(file)) {
      return blocks.some(
        (b) =>
          b.type === "file" &&
          b.mimeType === uploadMimeType(file) &&
          b.metadata?.filename === file.name,
      );
    }
    if (isSupportedUpload(file)) {
      return blocks.some(
        (b) =>
          b.type === "image" &&
          b.metadata?.name === file.name &&
          b.mimeType === file.type,
      );
    }
    return false;
  };

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    const fileArray = Array.from(files);
    const validFiles = fileArray.filter((file) =>
      isSupportedUpload(file),
    );
    const invalidFiles = fileArray.filter(
      (file) => !isSupportedUpload(file),
    );
    const duplicateFiles = validFiles.filter((file) =>
      isDuplicate(file, contentBlocks),
    );
    const uniqueFiles = validFiles.filter(
      (file) => !isDuplicate(file, contentBlocks),
    );

    if (invalidFiles.length > 0) {
      toast.error(
        UNSUPPORTED_FILE_TITLE, { description: UNSUPPORTED_FILE_BODY },
      );
    }
    if (duplicateFiles.length > 0) {
      toast.error(
        `Duplicate file(s) detected: ${duplicateFiles.map((f) => f.name).join(", ")}. Each file can only be uploaded once per message.`,
      );
    }

    const newBlocks = uniqueFiles.length
      ? await Promise.all(uniqueFiles.map(fileToContentBlock))
      : [];
    addBlocks(newBlocks);
    e.target.value = "";
  };

  // Drag and drop handlers
  useEffect(() => {
    if (!dropRef.current) return;

    // Global drag events with counter for robust dragOver state
    const handleWindowDragEnter = (e: DragEvent) => {
      if (e.dataTransfer?.types?.includes("Files")) {
        dragCounter.current += 1;
        setDragOver(true);
      }
    };
    const handleWindowDragLeave = (e: DragEvent) => {
      if (e.dataTransfer?.types?.includes("Files")) {
        dragCounter.current -= 1;
        if (dragCounter.current <= 0) {
          setDragOver(false);
          dragCounter.current = 0;
        }
      }
    };
    const handleWindowDrop = async (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter.current = 0;
      setDragOver(false);

      if (!e.dataTransfer) return;

      const files = Array.from(e.dataTransfer.files);
      const validFiles = files.filter((file) =>
        isSupportedUpload(file),
      );
      const invalidFiles = files.filter(
        (file) => !isSupportedUpload(file),
      );
      const duplicateFiles = validFiles.filter((file) =>
        isDuplicate(file, contentBlocks),
      );
      const uniqueFiles = validFiles.filter(
        (file) => !isDuplicate(file, contentBlocks),
      );

      if (invalidFiles.length > 0) {
        toast.error(
          UNSUPPORTED_FILE_TITLE, { description: UNSUPPORTED_FILE_BODY },
        );
      }
      if (duplicateFiles.length > 0) {
        toast.error(
          `Duplicate file(s) detected: ${duplicateFiles.map((f) => f.name).join(", ")}. Each file can only be uploaded once per message.`,
        );
      }

      const newBlocks = uniqueFiles.length
        ? await Promise.all(uniqueFiles.map(fileToContentBlock))
        : [];
      addBlocks(newBlocks);
    };
    const handleWindowDragEnd = (e: DragEvent) => {
      dragCounter.current = 0;
      setDragOver(false);
    };
    window.addEventListener("dragenter", handleWindowDragEnter);
    window.addEventListener("dragleave", handleWindowDragLeave);
    window.addEventListener("drop", handleWindowDrop);
    window.addEventListener("dragend", handleWindowDragEnd);

    // Prevent default browser behavior for dragover globally
    const handleWindowDragOver = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("dragover", handleWindowDragOver);

    // Remove element-specific drop event (handled globally)
    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(true);
    };
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(true);
    };
    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(false);
    };
    const element = dropRef.current;
    element.addEventListener("dragover", handleDragOver);
    element.addEventListener("dragenter", handleDragEnter);
    element.addEventListener("dragleave", handleDragLeave);

    return () => {
      element.removeEventListener("dragover", handleDragOver);
      element.removeEventListener("dragenter", handleDragEnter);
      element.removeEventListener("dragleave", handleDragLeave);
      window.removeEventListener("dragenter", handleWindowDragEnter);
      window.removeEventListener("dragleave", handleWindowDragLeave);
      window.removeEventListener("drop", handleWindowDrop);
      window.removeEventListener("dragend", handleWindowDragEnd);
      window.removeEventListener("dragover", handleWindowDragOver);
      dragCounter.current = 0;
    };
  }, [contentBlocks]);

  const removeBlock = (idx: number) => {
    setContentBlocks((prev) => prev.filter((_, i) => i !== idx));
  };

  const resetBlocks = () => setContentBlocks([]);

  /**
   * Handle paste event for files (images, PDFs)
   * Can be used as onPaste={handlePaste} on a textarea or input
   */
  const handlePaste = async (
    e: React.ClipboardEvent<HTMLTextAreaElement | HTMLInputElement>,
  ) => {
    const items = e.clipboardData.items;
    if (!items) return;
    const files: File[] = [];
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i];
      if (item.kind === "file") {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
    if (files.length === 0) {
      return;
    }
    e.preventDefault();
    const validFiles = files.filter((file) =>
      isSupportedUpload(file),
    );
    const invalidFiles = files.filter(
      (file) => !isSupportedUpload(file),
    );
    const isDuplicate = (file: File) => {
      if (isFileBlockUpload(file)) {
        return contentBlocks.some(
          (b) =>
            b.type === "file" &&
            b.mimeType === uploadMimeType(file) &&
            b.metadata?.filename === file.name,
        );
      }
      if (isSupportedUpload(file)) {
        return contentBlocks.some(
          (b) =>
            b.type === "image" &&
            b.metadata?.name === file.name &&
            b.mimeType === file.type,
        );
      }
      return false;
    };
    const duplicateFiles = validFiles.filter(isDuplicate);
    const uniqueFiles = validFiles.filter((file) => !isDuplicate(file));
    if (invalidFiles.length > 0) {
      toast.error(
        UNSUPPORTED_FILE_TITLE, { description: UNSUPPORTED_FILE_BODY },
      );
    }
    if (duplicateFiles.length > 0) {
      toast.error(
        `Duplicate file(s) detected: ${duplicateFiles.map((f) => f.name).join(", ")}. Each file can only be uploaded once per message.`,
      );
    }
    if (uniqueFiles.length > 0) {
      const newBlocks = await Promise.all(uniqueFiles.map(fileToContentBlock));
      addBlocks(newBlocks);
    }
  };

  return {
    contentBlocks,
    setContentBlocks,
    handleFileUpload,
    dropRef,
    removeBlock,
    resetBlocks,
    dragOver,
    handlePaste,
  };
}
