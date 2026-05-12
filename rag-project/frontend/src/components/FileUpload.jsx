import { useState, useRef, useCallback } from "react";
import { uploadFileLocally } from "../services/api";

const ACCEPT = ".pdf,.docx";
const MAX_SIZE = 20 * 1024 * 1024;

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function FileUpload({ onUploadSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState(null); // { type: "success"|"error", text }
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const validateFile = useCallback((file) => {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["pdf", "docx"].includes(ext)) {
      setMessage({ type: "error", text: "Only PDF and DOCX files are allowed." });
      return false;
    }
    if (file.size > MAX_SIZE) {
      setMessage({ type: "error", text: `File exceeds 20MB limit (${formatSize(file.size)}).` });
      return false;
    }
    return true;
  }, []);

  const handleFileSelect = (file) => {
    setMessage(null);
    if (file && validateFile(file)) {
      setSelectedFile(file);
    } else {
      setSelectedFile(null);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setMessage(null);

    try {
      const result = await uploadFileLocally(selectedFile);
      setMessage({ type: "success", text: `Uploaded "${result.filename}" successfully.` });
      onUploadSuccess(result);
      setSelectedFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || "Upload failed.";
      setMessage({ type: "error", text: detail });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-card">
      <h2>Upload Document</h2>

      <div
        className={`drop-zone ${dragActive ? "drop-zone--active" : ""}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <div className="drop-zone__icon">📄</div>
        <p className="drop-zone__text">
          Drag & drop a file here, or <span className="drop-zone__browse">browse</span>
        </p>
        <p className="drop-zone__hint">PDF and DOCX only, max 20MB</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="drop-zone__input"
          onChange={(e) => handleFileSelect(e.target.files[0])}
        />
      </div>

      {selectedFile && (
        <div className="selected-file">
          <span className="selected-file__name">{selectedFile.name}</span>
          <span className="selected-file__size">{formatSize(selectedFile.size)}</span>
        </div>
      )}

      {selectedFile && (
        <button
          className="upload-btn"
          onClick={handleUpload}
          disabled={uploading}
        >
          {uploading ? (
            <span className="spinner" />
          ) : (
            "Upload"
          )}
        </button>
      )}

      {message && (
        <div className={`message message--${message.type}`}>
          {message.text}
        </div>
      )}
    </div>
  );
}
