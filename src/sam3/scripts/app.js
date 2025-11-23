const form = document.getElementById("seg-form");
const imageInput = document.getElementById("image");

// Sidebar controls
const uploadGroup = document.getElementById("upload-group");
const thumbnailGroup = document.getElementById("thumbnail-group");
const removeImageBtn = document.getElementById("remove-image");
const sidebarThumb = document.getElementById("sidebar-thumb");
const openEditorBtn = document.getElementById("open-editor-btn");
let sidebarVideo = document.getElementById("sidebar-video");

// Inputs
const promptInput = document.getElementById("prompt");
const thresholdInput = document.getElementById("threshold");
const maskThresholdInput = document.getElementById("mask_threshold");
const thresholdValue = document.querySelector("[data-threshold-value]");
const maskThresholdValue = document.querySelector(
  "[data-mask-threshold-value]"
);

// Status + overlays
const statusEl = document.getElementById("status");
const submitBtn = document.getElementById("submit-btn");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingOverlayText = loadingOverlay?.querySelector("p");

// Modal + drawing canvas
const modal = document.getElementById("editor-modal");
const closeModalBtn = document.getElementById("close-modal");
const inputPreview = document.getElementById("input-preview");
const boxCanvas = document.getElementById("box-canvas");
const contextMenu = document.getElementById("box-context-menu");

// Box list
const boxList = document.getElementById("box-list");
const clearBoxesBtn = document.getElementById("clear-boxes");

// Results
const previewTabs = document.querySelectorAll("[data-preview-tab]");
const previewSlides = document.querySelectorAll("[data-preview-slide]");
const resultBoxCanvas = document.getElementById("result-box-canvas");
const masksEmptyState = document.getElementById("masks-empty");
const boxesEmptyState = document.getElementById("boxes-empty");

const statsEl = document.getElementById("generation-stats");
const videoProgressBar = document.getElementById("video-progress-bar");
const videoProgressFill = document.getElementById("video-progress-fill");
const videoProgressLabel = document.getElementById("video-progress-label");

const previewImages = {};
previewSlides.forEach((slide) => {
  const type = slide.dataset.previewSlide;
  const img = slide.querySelector("img[data-preview-image]");
  if (type && img) {
    previewImages[type] = img;
  }
});

const processUrl = form?.dataset?.processUrl || "/process";

let boxes = [];
let pendingBox = null;
let naturalSize = { width: null, height: null };
let isDrawing = false;
let startPoint = null;
let latestResultBoxes = [];
let activeRequestId = 0;
let currentImageSrc = "";
let currentFileType = "";
let currentAbortController = null;

const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));

const updateSliderLabels = () => {
  if (thresholdValue)
    thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
  if (maskThresholdValue)
    maskThresholdValue.textContent = Number(maskThresholdInput.value).toFixed(
      2
    );
};

const setStatus = (message, isError = false) => {
  if (!statusEl) return;
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#ef4444" : "var(--text-muted)";
};

const setLoading = (
  isLoading,
  {
    message = "Running SAM3 Model...",
    button = submitBtn,
    idleText = "Generate Segmentation",
    busyText = "Processing...",
  } = {}
) => {
  if (loadingOverlay && loadingOverlayText) {
    loadingOverlayText.textContent = message;
    loadingOverlay.classList.toggle("is-visible", isLoading);
  }

  if (!button) return;

  if (isLoading) {
    button.dataset.idleText =
      button.dataset.idleText || button.textContent || idleText;
    button.disabled = true;
    button.textContent = busyText;
  } else {
    button.disabled = false;
    const fallback = button.dataset.idleText || idleText;
    button.textContent = fallback;
  }
};

const showEmptyState = (el, show) => {
  if (!el) return;
  el.style.display = show ? "flex" : "none";
};

const clearMaskPreview = () => {
  const maskImg = previewImages["masks"];
  if (!maskImg) return;
  maskImg.removeAttribute("src");
  maskImg.style.display = "none";
};

const clearResultCanvas = () => {
  if (!resultBoxCanvas) return;
  const ctx = resultBoxCanvas.getContext("2d");
  ctx.clearRect(0, 0, resultBoxCanvas.width, resultBoxCanvas.height);
};

const resetResultState = () => {
  latestResultBoxes = [];
  clearMaskPreview();
  clearResultCanvas();
  showEmptyState(masksEmptyState, true);
};

const toggleUploadState = (hasImage) => {
  if (!uploadGroup || !thumbnailGroup) return;
  uploadGroup.style.display = hasImage ? "none" : "flex";
  thumbnailGroup.style.display = hasImage ? "block" : "none";
};

const resetBoxes = () => {
  boxes = [];
  renderBoxes();
  renderBoxList();
};

const resetImage = () => {
  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }
  currentImageSrc = "";
  currentFileType = "";
  if (promptInput) promptInput.value = "";
  if (statsEl) statsEl.textContent = "";
  if (videoProgressBar) videoProgressBar.style.display = "none";
  if (videoProgressFill) videoProgressFill.style.width = "0%";
  if (videoProgressLabel) videoProgressLabel.textContent = "";
  updateActiveMediaLabel();
  if (imageInput) imageInput.value = "";
  if (sidebarThumb) sidebarThumb.removeAttribute("src");
  if (sidebarThumb) sidebarThumb.style.display = "";
  if (sidebarVideo) {
    sidebarVideo.pause();
    sidebarVideo.removeAttribute("src");
    sidebarVideo.style.display = "none";
  }
  if (inputPreview) inputPreview.removeAttribute("src");
  naturalSize = { width: null, height: null };
  resetResultState();
  resetBoxes();

  if (previewImages["boxes"]) {
    previewImages["boxes"].removeAttribute("src");
    previewImages["boxes"].style.display = "none";
  }
  showEmptyState(boxesEmptyState, true);

  toggleUploadState(false);
  if (submitBtn) submitBtn.disabled = true;
  setStatus("Waiting for input...");
};

const syncCanvasSize = () => {
  if (!boxCanvas || !inputPreview) return;
  const rect = inputPreview.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  boxCanvas.width = rect.width;
  boxCanvas.height = rect.height;
  renderBoxes();
};

const syncResultCanvas = () => {
  if (!resultBoxCanvas) return;
  const refImg = previewImages["boxes"];
  if (!refImg || !refImg.clientWidth || !refImg.clientHeight) return;
  resultBoxCanvas.width = refImg.clientWidth;
  resultBoxCanvas.height = refImg.clientHeight;
};

const loadImage = (file) => {
  if (!file) return;
  resetResultState();

  if (currentFileType === "video") {
    if (!sidebarVideo) {
      sidebarVideo = document.createElement("video");
      sidebarVideo.id = "sidebar-video";
      sidebarVideo.controls = true;
      sidebarVideo.style.maxWidth = "100%";
      sidebarVideo.style.display = "block";
      if (sidebarThumb && sidebarThumb.parentNode) {
        sidebarThumb.parentNode.insertBefore(
          sidebarVideo,
          sidebarThumb.nextSibling
        );
      }
    }
    if (sidebarThumb) sidebarThumb.style.display = "none";
    sidebarVideo.style.display = "block";
    const videoUrl = URL.createObjectURL(file);
    sidebarVideo.src = videoUrl;
    sidebarVideo.load();

    sidebarVideo.onended = sidebarVideo.onpause = () => {
      URL.revokeObjectURL(videoUrl);
    };

    if (inputPreview) inputPreview.style.display = "none";
    setStatus("Ready.");
    showEmptyState(boxesEmptyState, false);
    toggleUploadState(true);
    if (submitBtn) submitBtn.disabled = false;
    return;
  } else {
    if (sidebarVideo) {
      sidebarVideo.pause();
      sidebarVideo.removeAttribute("src");
      sidebarVideo.style.display = "none";
    }
    if (sidebarThumb) sidebarThumb.style.display = "";
    if (inputPreview) inputPreview.style.display = "";
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    currentImageSrc = event.target.result;
    if (sidebarThumb) sidebarThumb.src = currentImageSrc;
    if (inputPreview) inputPreview.src = currentImageSrc;
    if (previewImages["boxes"]) {
      previewImages["boxes"].src = currentImageSrc;
      previewImages["boxes"].style.display = "block";
    }

    showEmptyState(boxesEmptyState, false);
    toggleUploadState(true);
    setStatus("Ready.");
  };
  reader.readAsDataURL(file);
};

if (imageInput) {
  imageInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type.startsWith("image/")) {
        currentFileType = "image";
      } else if (file.type.startsWith("video/")) {
        currentFileType = "video";
      } else {
        currentFileType = "";
      }
      updateActiveMediaLabel();
      updateDrawBoxesOverlay();
      updateHelperText();
      loadImage(file);

      if (file.type.startsWith("video/")) {
        if (statsEl) statsEl.textContent = "Loading video metadata...";
        fetchVideoMetadata(file).then((meta) => {
          if (statsEl) {
            statsEl.textContent =
              `Video length: ${meta.duration.toFixed(2)}s | ` +
              `Total frames: ${meta.total_frames} | ` +
              `fps: ${meta.fps.toFixed(2)}`;
          }
        });
      }
    }
  });
}

if (removeImageBtn) removeImageBtn.addEventListener("click", resetImage);

if (inputPreview) {
  inputPreview.onload = () => {
    naturalSize = {
      width: inputPreview.naturalWidth,
      height: inputPreview.naturalHeight,
    };
    if (submitBtn) submitBtn.disabled = !naturalSize.width;
    resetBoxes();
    syncCanvasSize();
    setTimeout(syncResultCanvas, 50);
  };
}

const openModal = () => {
  if (!modal || !currentImageSrc) return;
  modal.classList.add("is-open");
  setTimeout(syncCanvasSize, 50);
};

const closeModal = () => {
  if (!modal) return;
  modal.classList.remove("is-open");
  hideContextMenu();
  cancelPendingBox();
};

if (openEditorBtn) openEditorBtn.addEventListener("click", openModal);
if (closeModalBtn) closeModalBtn.addEventListener("click", closeModal);

const normalizedPoint = (event) => {
  if (!boxCanvas) return { x: 0, y: 0 };
  const rect = boxCanvas.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width),
    y: clamp((event.clientY - rect.top) / rect.height),
  };
};

const renderBoxes = (currentDrag = null) => {
  if (!boxCanvas) return;
  const ctx = boxCanvas.getContext("2d");
  ctx.clearRect(0, 0, boxCanvas.width, boxCanvas.height);

  const drawBox = (box, isPending = false) => {
    const w = boxCanvas.width;
    const h = boxCanvas.height;
    const x1 = box.x1 * w;
    const y1 = box.y1 * h;
    const ww = (box.x2 - box.x1) * w;
    const hh = (box.y2 - box.y1) * h;

    ctx.lineWidth = 2;
    if (isPending) {
      ctx.strokeStyle = "#fff";
      ctx.setLineDash([5, 3]);
    } else {
      const isNeg = Number(box.label) === 0;
      ctx.strokeStyle = isNeg ? "#ef4444" : "#22c55e";
      ctx.setLineDash(isNeg ? [4, 4] : []);
      ctx.fillStyle = isNeg
        ? "rgba(239, 68, 68, 0.15)"
        : "rgba(34, 197, 94, 0.15)";
      ctx.fillRect(x1, y1, ww, hh);
    }
    ctx.strokeRect(x1, y1, ww, hh);
    ctx.setLineDash([]);
  };

  boxes.forEach((b) => drawBox(b));
  if (currentDrag) {
    const box = {
      x1: Math.min(startPoint.x, currentDrag.x),
      y1: Math.min(startPoint.y, currentDrag.y),
      x2: Math.max(startPoint.x, currentDrag.x),
      y2: Math.max(startPoint.y, currentDrag.y),
    };
    drawBox(box, true);
  }
};

const startDraw = (e) => {
  if (!boxCanvas) return;
  hideContextMenu();
  isDrawing = true;
  startPoint = normalizedPoint(e);
  boxCanvas.setPointerCapture(e.pointerId);
};

const moveDraw = (e) => {
  if (!isDrawing) return;
  const current = normalizedPoint(e);
  renderBoxes(current);
};

const endDraw = (e) => {
  if (!isDrawing) return;
  isDrawing = false;
  boxCanvas.releasePointerCapture(e.pointerId);
  const end = normalizedPoint(e);
  pendingBox = {
    x1: Math.min(startPoint.x, end.x),
    y1: Math.min(startPoint.y, end.y),
    x2: Math.max(startPoint.x, end.x),
    y2: Math.max(startPoint.y, end.y),
  };
  if (Math.abs(pendingBox.x2 - pendingBox.x1) < 0.01) {
    pendingBox = null;
    return;
  }
  showContextMenu(e.clientX, e.clientY);
};

const showContextMenu = (x, y) => {
  if (!contextMenu) return;
  const menuWidth = 200;
  const menuHeight = 80;
  let posX = x + 10;
  let posY = y + 10;
  if (posX + menuWidth > window.innerWidth) posX = x - menuWidth - 10;
  if (posY + menuHeight > window.innerHeight) posY = y - menuHeight - 10;
  contextMenu.style.left = `${posX}px`;
  contextMenu.style.top = `${posY}px`;
  contextMenu.classList.add("is-visible");
};

const hideContextMenu = () => {
  if (!contextMenu) return;
  contextMenu.classList.remove("is-visible");
};

const cancelPendingBox = () => {
  pendingBox = null;
  renderBoxes();
};

if (contextMenu) {
  contextMenu.addEventListener("click", (e) => {
    if (e.target.tagName !== "BUTTON") return;
    const label = Number(e.target.dataset.choice);
    if (pendingBox) {
      boxes.push({ ...pendingBox, label });
      renderBoxes();
      renderBoxList();
    }
    hideContextMenu();
    pendingBox = null;
  });
}

if (boxCanvas) {
  boxCanvas.addEventListener("pointerdown", startDraw);
  boxCanvas.addEventListener("pointermove", moveDraw);
  boxCanvas.addEventListener("pointerup", endDraw);
}

const renderBoxList = () => {
  if (!boxList) return;
  if (!boxes.length) {
    boxList.innerHTML = '<p class="muted-text">No boxes added.</p>';
    return;
  }

  boxList.innerHTML = "";
  const w = naturalSize.width || 1;
  const h = naturalSize.height || 1;
  boxes.forEach((box, idx) => {
    const el = document.createElement("div");
    el.className = "box-chip";
    el.dataset.type = box.label;
    const typeText = box.label === 1 ? "POS" : "NEG";
    const coordsText = `${Math.round(box.x1 * w)}, ${Math.round(box.y1 * h)}`;
    el.innerHTML = `<span><b>${typeText}</b> <span style="opacity:0.7; font-size:11px; margin-left:4px">(${coordsText})</span></span>
      <button type="button" class="box-chip__remove">×</button>`;
    el.querySelector("button").onclick = () => {
      boxes.splice(idx, 1);
      renderBoxes();
      renderBoxList();
    };
    boxList.appendChild(el);
  });
};

if (clearBoxesBtn) clearBoxesBtn.addEventListener("click", resetBoxes);

const drawResultBoxes = () => {
  if (!resultBoxCanvas) return;
  const ctx = resultBoxCanvas.getContext("2d");
  ctx.clearRect(0, 0, resultBoxCanvas.width, resultBoxCanvas.height);

  if (!latestResultBoxes.length || !naturalSize.width || !naturalSize.height) {
    return;
  }

  syncResultCanvas();

  const scaleX = resultBoxCanvas.width / naturalSize.width;
  const scaleY = resultBoxCanvas.height / naturalSize.height;

  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(99, 102, 241, 0.9)";
  ctx.shadowColor = "rgba(99, 102, 241, 0.4)";
  ctx.shadowBlur = 6;

  latestResultBoxes.forEach((box) => {
    const [x1, y1, x2, y2] = box;
    const rX = x1 * scaleX;
    const rY = y1 * scaleY;
    const rW = (x2 - x1) * scaleX;
    const rH = (y2 - y1) * scaleY;
    ctx.strokeRect(rX, rY, rW, rH);
  });

  ctx.shadowBlur = 0;
};

const applyResult = (data) => {
  resetResultState();

  if (Array.isArray(data.frames)) {
    let idx = 0;
    const frames = data.frames;
    const masksImg = previewImages["masks"];
    if (!masksImg) return;
    showEmptyState(masksEmptyState, false);

    const totalFrames = data.total_frames || frames.length;

    if (videoProgressBar) videoProgressBar.style.display = "block";

    if (window._sam3VideoAnim) clearInterval(window._sam3VideoAnim);

    window._sam3VideoAnim = setInterval(() => {
      const frame = frames[idx % frames.length];
      if (frame && frame.image) {
        masksImg.src = `data:image/png;base64,${frame.image}`;
        masksImg.style.display = "block";
      }
      idx++;

      if (videoProgressFill && videoProgressLabel) {
        const percent = Math.min(100, (idx / totalFrames) * 100);
        videoProgressFill.style.width = percent + "%";
        videoProgressLabel.textContent = `Frame ${Math.min(
          idx,
          totalFrames
        )} of ${totalFrames}`;
      }

      if (idx >= frames.length) {
        idx = 0;
        setStatus(
          `Processed ${frames.length} of ${totalFrames} video frames.`
        );
      }
    }, 150);

    setStatus(`Processing ${frames.length} of ${totalFrames} video frames...`);
    return;
  }

  const masksImg = previewImages["masks"];
  if (data.image && masksImg) {
    masksImg.src = `data:image/png;base64,${data.image}`;
    masksImg.style.display = "block";
    showEmptyState(masksEmptyState, false);
  } else {
    showEmptyState(masksEmptyState, true);
  }

  if (Array.isArray(data.boxes)) {
    latestResultBoxes = data.boxes;
  } else if (Array.isArray(data.pred_boxes)) {
    latestResultBoxes = data.pred_boxes;
  } else {
    latestResultBoxes = [];
  }

  if (latestResultBoxes.length) {
    showEmptyState(boxesEmptyState, false);
  } else {
    showEmptyState(boxesEmptyState, true);
  }

  drawResultBoxes();
  setActiveTab("masks");
  setStatus(
    data.count ? `Found ${data.count} regions.` : "Inference complete."
  );
};

if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (currentAbortController) {
      currentAbortController.abort();
    }
    currentAbortController = new AbortController();
    const signal = currentAbortController.signal;

    if (!imageInput?.files?.length) {
      setStatus("Please select an image.", true);
      return;
    }

    const requestId = ++activeRequestId;
    resetResultState();
    setLoading(true);
    setStatus("Running model...");
    const startTime = performance.now();

    const isVideo = currentFileType === "video";
    const formData = new FormData();
    const fileFieldName = isVideo ? "video" : "image";
    formData.append(fileFieldName, imageInput.files[0]);

    formData.append(fileFieldName, imageInput.files[0]);
    formData.append("prompt", promptInput?.value || "");
    formData.append("threshold", thresholdInput ? thresholdInput.value : "0.5");
    formData.append(
      "mask_threshold",
      maskThresholdInput ? maskThresholdInput.value : "0.5"
    );

    if (boxes.length && naturalSize.width && naturalSize.height) {
      const w = naturalSize.width;
      const h = naturalSize.height;
      const payload = {
        boxes: boxes.map((b) => [b.x1 * w, b.y1 * h, b.x2 * w, b.y2 * h]),
        labels: boxes.map((b) => b.label),
      };
      formData.append("boxes", JSON.stringify(payload));
    }

    const url =
      (form?.dataset?.processUrl && !isVideo
        ? form.dataset.processUrl
        : null) || (isVideo ? "/process-video" : "/process");

    try {
      const res = await fetch(url, {
        method: "POST",
        body: formData,
        cache: "no-store",
        signal,
      });
      const data = await res.json();

      const endTime = performance.now();
      const elapsed = (endTime - startTime) / 1000;

      if (statsEl) {
        const genTimeText = `Generation time: ${elapsed.toFixed(2)}s`;
        statsEl.textContent += (statsEl.textContent ? "\n" : "") + genTimeText;
      }

      if (requestId !== activeRequestId) return;
      if (data.status === "ok") {
        applyResult(data);
      } else {
        throw new Error(data.detail || "Backend error");
      }
    } catch (err) {
      console.error(err);
      if (requestId !== activeRequestId) return;
    } finally {
      if (requestId !== activeRequestId) return;
      setLoading(false);
    }
  });
}

const setActiveTab = (id) => {
  previewTabs.forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.previewTab === id);
  });
  previewSlides.forEach((slide) => {
    slide.classList.toggle("is-visible", slide.dataset.previewSlide === id);
  });

  if (id === "boxes") {
    setTimeout(drawResultBoxes, 20);
  }
};

previewTabs.forEach((tab) => {
  tab.addEventListener("click", () => setActiveTab(tab.dataset.previewTab));
});

window.addEventListener("resize", () => {
  if (modal?.classList.contains("is-open")) syncCanvasSize();
  syncResultCanvas();
  drawResultBoxes();
});

updateSliderLabels();
if (thresholdInput)
  thresholdInput.addEventListener("input", updateSliderLabels);
if (maskThresholdInput)
  maskThresholdInput.addEventListener("input", updateSliderLabels);

function updateActiveMediaLabel() {
  const label = document.getElementById("active-media-label");
  const removeBtn = document.getElementById("remove-image");
  if (!label || !removeBtn) return;
  if (currentFileType === "video") {
    label.textContent = "Active Video";
    removeBtn.textContent = "Remove Video";
  } else {
    label.textContent = "Active Image";
    removeBtn.textContent = "Remove Image";
  }
}

function updateDrawBoxesOverlay() {
  const overlay = document.getElementById("draw-boxes-overlay");
  if (!overlay) return;
  if (currentFileType === "video") {
    overlay.style.display = "none";
  } else {
    overlay.style.display = "";
  }
}

function updateHelperText() {
  const helperText = document.getElementById("helper-text");
  if (!helperText) return;
  helperText.style.display = currentFileType === "video" ? "none" : "";
}

async function fetchVideoMetadata(file) {
  const formData = new FormData();
  formData.append("video", file);
  const res = await fetch("/video-metadata", {
    method: "POST",
    body: formData,
  });
  return res.json();
}
