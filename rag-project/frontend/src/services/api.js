import axios from "axios";

export async function uploadFileLocally(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await axios.post("/api/upload/local", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
}
