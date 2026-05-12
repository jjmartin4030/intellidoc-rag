function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(isoString) {
  return new Date(isoString).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function DocumentList({ documents }) {
  if (!documents.length) {
    return (
      <div className="doclist-card">
        <h2>Uploaded Documents</h2>
        <div className="doclist-empty">
          <p>No documents uploaded yet</p>
        </div>
      </div>
    );
  }

  return (
    <div className="doclist-card">
      <h2>Uploaded Documents</h2>
      <ul className="doclist">
        {documents.map((doc, idx) => (
          <li key={idx} className="doclist__item">
            <div className="doclist__info">
              <span className="doclist__name">{doc.filename}</span>
              <span className="doclist__meta">
                {formatSize(doc.size_bytes)} · {formatTime(doc.saved_at)}
              </span>
            </div>
            <span className={`badge badge--${doc.file_type}`}>
              {doc.file_type.toUpperCase()}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
