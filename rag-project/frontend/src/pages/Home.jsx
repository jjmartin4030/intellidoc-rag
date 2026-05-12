import { useState } from "react";
import FileUpload from "../components/FileUpload";
import DocumentList from "../components/DocumentList";

export default function Home() {
  const [documents, setDocuments] = useState([]);

  const handleUploadSuccess = (doc) => {
    setDocuments((prev) => [doc, ...prev]);
  };

  return (
    <div className="home">
      <div className="home__grid">
        <FileUpload onUploadSuccess={handleUploadSuccess} />
        <DocumentList documents={documents} />
      </div>
    </div>
  );
}
