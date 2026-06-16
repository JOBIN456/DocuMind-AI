 const fileInput  = document.getElementById('fileInput');
    const uploadZone = document.getElementById('uploadZone');
    const fileInfo   = document.getElementById('fileInfo');
    const fileName   = document.getElementById('fileName');
    const removeFile = document.getElementById('removeFile');
    const btnUpload  = document.getElementById('btnUpload');
    const btnChat    = document.getElementById('btnChat');
    const step1      = document.getElementById('step1');
    const step2      = document.getElementById('step2');

    /* Open file picker */
    uploadZone.addEventListener('click', () => fileInput.click());
    btnUpload.addEventListener('click', () => fileInput.click());

    /* File chosen via picker */
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) setFile(fileInput.files[0]);
    });

    /* Drag & drop */
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file && file.type === 'application/pdf') setFile(file);
      else alert('Please drop a PDF file.');
    });

    /* Show file info */
    function setFile(file) {
      fileName.textContent = file.name;
      uploadZone.classList.add('hidden');
      fileInfo.classList.add('visible');
      step1.classList.remove('active');
      step2.classList.add('active');
      btnUpload.innerHTML = '<i class="ti ti-replace"></i> Replace PDF';
    }

    /* Remove file */
    removeFile.addEventListener('click', () => {
      fileInput.value = '';
      fileInfo.classList.remove('visible');
      uploadZone.classList.remove('hidden');
      step1.classList.add('active');
      step2.classList.remove('active');
      btnUpload.innerHTML = '<i class="ti ti-upload"></i> Upload PDF';
    });

    let collectionName = '';  // ← stores collection name after upload

/* Upload API call — triggered when file is selected */
async function uploadToAPI(file) {
  const formData = new FormData();
  formData.append('file', file);

  btnUpload.disabled = true;
  btnUpload.innerHTML = '<i class="ti ti-loader"></i> Uploading…';

  try {
    const res  = await fetch('http://127.0.0.1:8000/pdf/ingest', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      alert('Upload failed: ' + (data.detail || res.statusText));
      resetUpload();
      return;
    }

    collectionName = data.collection;   // ← save for chat page
    btnUpload.innerHTML = '<i class="ti ti-circle-check"></i> Uploaded';

  } catch (err) {
    alert('Cannot reach the server. Make sure the API is running correctly');
    resetUpload();
  }

  btnUpload.disabled = false;
}

/* Reset upload state on error */
function resetUpload() {
  fileInput.value = '';
  fileInfo.classList.remove('visible');
  uploadZone.classList.remove('hidden');
  step1.classList.add('active');
  step2.classList.remove('active');
  btnUpload.disabled = false;
  btnUpload.innerHTML = '<i class="ti ti-upload"></i> Upload PDF';
}

/* Show file info — also triggers upload */
function setFile(file) {
  fileName.textContent = file.name;
  uploadZone.classList.add('hidden');
  fileInfo.classList.add('visible');
  step1.classList.remove('active');
  step2.classList.add('active');
  uploadToAPI(file);   // ← call API as soon as file is picked
}

/* Chat button — go to chat page with collection name */
btnChat.addEventListener('click', () => {
  const hasFile = fileInfo.classList.contains('visible');
  if (!hasFile) {
    alert('Please upload a PDF first.');
    return;
  }
  if (!collectionName) {
    alert('Still uploading, please wait…');
    return;
  }
  // pass collection name to chat page via URL param
  window.location.href = `/chatpage?pdf=${encodeURIComponent(collectionName)}`;

});