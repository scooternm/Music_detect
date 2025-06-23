async function handleUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
  
    const response = await fetch("http://127.0.0.1:8000/uploadfile/", {
      method: "POST",
      body: formData,
    });
  
    const data = await response.json();
    console.log("Tempo data:", data);
  }  