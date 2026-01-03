const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const video = document.getElementById('camera-feed');
const mainBox = document.getElementById('mainBox');
const camToggleBtn = document.getElementById('camToggleBtn');
const eraserBtn = document.getElementById('eraserBtn');
const fileInput = document.getElementById('fileInput');
const digitDisplay = document.getElementById('digit-display');
const confDetails = document.getElementById('conf-details');

const voiceBtn = document.getElementById('voiceBtn');
const voiceStatus = document.getElementById('voice-status');
let recognition = null;
let isVoiceActive = false;

let model = null;
let isDrawing = false;
let isEraser = false;
let cameraStream = null;
let realtimeInterval = null;
let lastX = 0;
let lastY = 0;

// --- 1. 系統初始化與模型載入 (針對截圖報錯進行最終修正) ---
async function init() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();
    addGalaxyEffects();

    try {
        confDetails.innerText = "🌌 正在啟動銀河 AI 引擎...";
        
        // 強制切換至 CPU 模式，徹底解決 WebGL 不支援報錯
        await tf.setBackend('cpu');
        await tf.ready();
        console.log("當前運行後端:", tf.getBackend());

        // 核心修正：移除會導致 'producer' 錯誤的 loadGraphModel 嘗試
        // 如果你的模型是從 Keras 轉換而來，必須使用 loadLayersModel
        try {
            // 加入抗快取參數，確保每次載入的都是最新的 GitHub 檔案
            const modelUrl = `tfjs_model/model.json?t=${Date.now()}`;
            model = await tf.loadLayersModel(modelUrl);
            console.log("✅ 模型載入成功 (LayersModel)");
            confDetails.innerText = "🚀 系統就緒，請開始在星域書寫";
        } catch (err) {
            console.error("載入失敗:", err);
            confDetails.innerText = "❌ 模型載入失敗：請確認 tfjs_model 資料夾內是否有 bin 檔";
        }
    } catch (err) {
        confDetails.innerText = "❌ 系統初始化失敗";
        console.error(err);
    }
}

// --- 2. 影像處理與辨識邏輯 (完全保留，不動任何辨識部分) ---

function advancedPreprocess(roiCanvas) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(roiCanvas, 1);
        tensor = tensor.toFloat().div(tf.scalar(255.0));
        tensor = tf.image.resizeBilinear(tensor, [28, 28]);
        return tensor.expandDims(0);
    });
}

async function predict() {
    if (!model) return;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const boxes = findDigitBoxes(imageData);
    
    let finalRes = "";
    let details = [];
    let validBoxes = [];

    for (let box of boxes) {
        const { x, y, w, h, area } = box;
        const MIN_AREA = cameraStream ? 500 : 150;
        if (area < MIN_AREA) continue;
        
        const aspectRatio = w / h;
        if (aspectRatio > 2.5 || aspectRatio < 0.15) continue;

        const roiCanvas = document.createElement('canvas');
        roiCanvas.width = w;
        roiCanvas.height = h;
        const roiCtx = roiCanvas.getContext('2d');
        roiCtx.drawImage(canvas, x, y, w, h, 0, 0, w, h);

        if (w > h * 1.3) {
            const splitX = Math.floor(w / 2);
            const subWidths = [splitX, w - splitX];
            const subXOffsets = [0, splitX];

            for (let i = 0; i < 2; i++) {
                const subCanvas = document.createElement('canvas');
                subCanvas.width = subWidths[i];
                subCanvas.height = h;
                const subCtx = subCanvas.getContext('2d');
                subCtx.drawImage(roiCanvas, subXOffsets[i], 0, subWidths[i], h, 0, 0, subWidths[i], h);
                
                const input = advancedPreprocess(subCanvas);
                const pred = model.predict(input);
                const score = await pred.data();
                const digit = pred.argMax(-1).dataSync()[0];
                const conf = Math.max(...score);

                if (conf > 0.8) {
                    finalRes += digit.toString();
                    details.push({ digit, conf: (conf * 100).toFixed(1) + "%" });
                }
                input.dispose(); pred.dispose();
            }
            continue;
        }

        const input = advancedPreprocess(roiCanvas);
        const pred = model.predict(input);
        const score = await pred.data();
        const digit = pred.argMax(-1).dataSync()[0];
        const conf = Math.max(...score);

        if (conf > 0.85) {
            finalRes += digit.toString();
            details.push({ digit, conf: (conf * 100).toFixed(1) + "%" });
            validBoxes.push(box);
        }
        input.dispose(); pred.dispose();
    }

    digitDisplay.innerText = finalRes || "---";
    updateDetails(details);

    if (cameraStream) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        validBoxes.forEach((box, i) => {
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 3;
            ctx.strokeRect(box.x, box.y, box.w, box.h);
            ctx.fillStyle = "#00FF00";
            ctx.font = "bold 24px Arial";
            ctx.fillText(details[i].digit, box.x, box.y - 5);
        });
        updatePen();
    }

    if (finalRes !== "") addVisualFeedback("#2ecc71");
}

function findDigitBoxes(imageData) {
    const { data, width, height } = imageData;
    const visited = new Uint8Array(width * height);
    const boxes = [];

    for (let y = 0; y < height; y += 4) {
        for (let x = 0; x < width; x += 4) {
            const idx = (y * width + x);
            if (!visited[idx] && data[idx * 4] > 100) {
                let minX = x, maxX = x, minY = y, maxY = y, count = 0;
                const queue = [[x, y]];
                visited[idx] = 1;

                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    count++;
                    if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
                    if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;

                    const neighbors = [[cx+8, cy], [cx-8, cy], [cx, cy+8], [cx, cy-8]];
                    for (let [nx, ny] of neighbors) {
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = (ny * width + nx);
                            if (!visited[nIdx] && data[nIdx * 4] > 100) {
                                visited[nIdx] = 1;
                                queue.push([nx, ny]);
                            }
                        }
                    }
                }
                boxes.push({ x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1, area: count * 16 });
            }
        }
    }
    return boxes.sort((a, b) => a.x - b.x);
}

// --- 3. 其他 UI 輔助邏輯 (保留原樣) ---

function addGalaxyEffects() {
    setTimeout(() => {
        if (!cameraStream) {
            ctx.fillStyle = "rgba(163, 217, 255, 0.3)";
            ctx.beginPath(); ctx.arc(650, 20, 3, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(30, 300, 2, 0, Math.PI * 2); ctx.fill();
            updatePen();
        }
    }, 500);
}

function updatePen() {
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    if (isEraser) { ctx.strokeStyle = "black"; ctx.lineWidth = 40; }
    else { ctx.strokeStyle = "white"; ctx.lineWidth = 15; }
}

function toggleEraser() {
    isEraser = !isEraser;
    eraserBtn.innerText = isEraser ? "橡皮擦：開啟" : "橡皮擦：關閉";
    eraserBtn.classList.toggle('eraser-active', isEraser);
    updatePen();
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!cameraStream) { ctx.fillStyle = "black"; ctx.fillRect(0, 0, canvas.width, canvas.height); }
    digitDisplay.innerText = "---";
    addGalaxyEffects();
}

function addVisualFeedback(color) {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
        const originalBoxShadow = btn.style.boxShadow;
        btn.style.boxShadow = `0 0 20px ${color}`;
        setTimeout(() => { btn.style.boxShadow = originalBoxShadow; }, 300);
    });
}

async function toggleCamera() {
    if (cameraStream) {
        stopCamera();
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment", width: 1280, height: 720 },
                audio: false
            });
            video.srcObject = cameraStream;
            video.style.display = "block";
            mainBox.classList.add('cam-active');
            camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 關閉鏡頭';
            realtimeInterval = setInterval(() => predict(), 400);
            clearCanvas();
        } catch (err) { alert("鏡頭啟動失敗: " + err); }
    }
}

function stopCamera() {
    if (cameraStream) { cameraStream.getTracks().forEach(track => track.stop()); cameraStream = null; }
    if (realtimeInterval) clearInterval(realtimeInterval);
    video.style.display = "none";
    mainBox.classList.remove('cam-active');
    camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 開啟鏡頭';
    init();
}

// --- 4. 事件監聽與語音 (保留原樣) ---

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('touchstart', (e) => { if (e.touches.length === 1) startDrawing(e); });
canvas.addEventListener('touchmove', (e) => { if (e.touches.length === 1) draw(e); });
canvas.addEventListener('touchend', stopDrawing);

function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    let x, y;
    if (e.type.includes('touch')) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }
    return { x, y };
}

function startDrawing(e) {
    e.preventDefault(); isDrawing = true;
    const { x, y } = getCanvasCoordinates(e);
    ctx.beginPath(); ctx.moveTo(x, y);
}

function draw(e) {
    e.preventDefault(); if (!isDrawing) return;
    const { x, y } = getCanvasCoordinates(e);
    ctx.lineTo(x, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y);
}

function stopDrawing() {
    if (isDrawing) { isDrawing = false; ctx.beginPath(); if (!cameraStream) setTimeout(() => predict(), 100); }
}

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { voiceBtn.style.display = 'none'; return; }
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-TW';
    recognition.onstart = () => { isVoiceActive = true; updateVoiceButton(); };
    recognition.onend = () => { isVoiceActive = false; updateVoiceButton(); };
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript.trim();
        if (transcript.includes('清除')) clearCanvas();
        else if (transcript.includes('辨識')) predict();
    };
}

function toggleVoice() { if (isVoiceActive) recognition.stop(); else recognition.start(); }
function updateVoiceButton() {
    voiceBtn.innerHTML = isVoiceActive ? '<span class="btn-icon">🌌</span> 聽取中...' : '<span class="btn-icon">🌌</span> 語音輸入';
    voiceBtn.classList.toggle('voice-active', isVoiceActive);
}

function triggerFile() { fileInput.click(); }
function handleFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            ctx.fillStyle = "black"; ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 50, 50, canvas.width - 100, canvas.height - 100);
            predict();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function updateDetails(data) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (data.length === 0) html += "等待有效數字入鏡...";
    else data.forEach((item, i) => { html += `數字 ${i + 1}: <b style="color:#a3d9ff">${item.digit}</b> (信心度: ${item.conf})<br>`; });
    confDetails.innerHTML = html;
}

init();
