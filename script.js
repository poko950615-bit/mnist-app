/**
 * 🌌 銀河手寫數字辨識系統 - 終極整合版 (修復載入 + 完整功能)
 */

// --- 1. 元素選取與變數設定 ---
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const video = document.getElementById('camera-feed');
const digitDisplay = document.getElementById('digit-display');
const confDetails = document.getElementById('conf-details');

// 按鈕們
const eraserBtn = document.getElementById('eraserBtn');
const camToggleBtn = document.getElementById('camToggleBtn');
const voiceBtn = document.getElementById('voiceBtn');
const fileInput = document.getElementById('fileInput');

let model = null;
let isDrawing = false;
let isEraser = false;
let cameraStream = null;
let realtimeInterval = null;
let recognition = null;
let isVoiceActive = false;

// --- 🛠️ 模型修復載入器 (解決 Keras v3 相容性) ---
class PatchModelLoader {
    constructor(url) { this.url = url; }
    async load() {
        const loader = tf.io.browserHTTPRequest(this.url);
        const artifacts = await loader.load();
        
        // 修復 A: 注入缺失的 InputLayer 形狀
        const traverseAndPatch = (obj) => {
            if (!obj || typeof obj !== 'object') return;
            if (obj.class_name === 'InputLayer' && obj.config) {
                const cfg = obj.config;
                if (!cfg.batchInputShape && !cfg.batch_input_shape) {
                    cfg.batchInputShape = [null, 28, 28, 1];
                }
            }
            if (Array.isArray(obj)) obj.forEach(item => traverseAndPatch(item));
            else Object.keys(obj).forEach(key => traverseAndPatch(obj[key]));
        };
        if (artifacts.modelTopology) traverseAndPatch(artifacts.modelTopology);

        // 修復 B: 移除權重名稱中的 'sequential/' 前綴
        if (artifacts.weightSpecs) {
            artifacts.weightSpecs.forEach(spec => {
                if (spec.name.includes('sequential/')) {
                    spec.name = spec.name.replace('sequential/', '');
                }
            });
        }
        return artifacts;
    }
}

// --- 2. 系統初始化 ---
async function init() {
    // 初始化畫布
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();

    const modelUrl = `tfjs_model/model.json?t=${Date.now()}`;

    try {
        confDetails.innerText = "🌌 正在啟動銀河辨識引擎...";
        
        // 優先嘗試使用 CPU 以確保穩定，若想提升效能可試著註解掉下一行
        await tf.setBackend('cpu');
        await tf.ready();

        model = await tf.loadLayersModel(new PatchModelLoader(modelUrl));
        
        console.log("✅ 模型載入成功！");
        confDetails.innerText = "🚀 系統就緒，請開始在星域書寫";
        
        // 模型暖身
        tf.tidy(() => model.predict(tf.zeros([1, 28, 28, 1])));

    } catch (err) {
        console.error("載入失敗:", err);
        confDetails.innerHTML = `<span style="color: #ff4d4d">❌ 錯誤: ${err.message}</span>`;
    }
}

// --- 3. 影像處理核心 (辨識功能就在這) ---
function preprocess(roiCanvas) {
    return tf.tidy(() => {
        // 將畫布轉為 Tensor (灰階 1 channel)
        let tensor = tf.browser.fromPixels(roiCanvas, 1);
        // 標準化 0~1
        tensor = tensor.toFloat().div(tf.scalar(255.0));
        // 縮放至 MNIST 標準 28x28
        tensor = tf.image.resizeBilinear(tensor, [28, 28]);
        // 增加 batch 維度 [1, 28, 28, 1]
        return tensor.expandDims(0);
    });
}

async function predict() {
    if (!model) return;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const boxes = findDigitBoxes(imageData);
    
    let finalRes = "";
    let details = [];

    for (let box of boxes) {
        if (box.area < 100) continue; // 過濾掉太小的雜點

        const roiCanvas = document.createElement('canvas');
        roiCanvas.width = box.w; roiCanvas.height = box.h;
        roiCanvas.getContext('2d').drawImage(canvas, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);

        const input = preprocess(roiCanvas);
        const pred = model.predict(input);
        const score = await pred.data();
        const digit = pred.argMax(-1).dataSync()[0];
        const conf = Math.max(...score);

        // 如果信心度夠高才顯示
        if (conf > 0.7) {
            finalRes += digit.toString();
            details.push({ digit, conf: (conf * 100).toFixed(1) + "%" });
        }
        input.dispose(); pred.dispose();
    }

    digitDisplay.innerText = finalRes || "---";
    updateDetails(details);
}

// 尋找數字區域 (連通域算法)
function findDigitBoxes(imageData) {
    const { data, width, height } = imageData;
    const visited = new Uint8Array(width * height);
    const boxes = [];

    for (let y = 0; y < height; y += 4) {
        for (let x = 0; x < width; x += 4) {
            const idx = y * width + x;
            if (!visited[idx] && data[idx * 4] > 80) { // 偵測白色像素
                let minX = x, maxX = x, minY = y, maxY = y, count = 0;
                let queue = [[x, y]];
                visited[idx] = 1;

                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    count++;
                    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);

                    [[cx+8, cy], [cx-8, cy], [cx, cy+8], [cx, cy-8]].forEach(([nx, ny]) => {
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            if (!visited[nIdx] && data[nIdx * 4] > 80) {
                                visited[nIdx] = 1; queue.push([nx, ny]);
                            }
                        }
                    });
                }
                boxes.push({ x: minX, y: minY, w: maxX-minX+1, h: maxY-minY+1, area: count });
            }
        }
    }
    return boxes.sort((a, b) => a.x - b.x);
}

// --- 4. UI 互動功能 ---
function updatePen() {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (isEraser) {
        ctx.strokeStyle = "black";
        ctx.lineWidth = 40;
    } else {
        ctx.strokeStyle = "white";
        ctx.lineWidth = 15; // 若辨識不準，可調整畫筆粗細
    }
}

function toggleEraser() {
    isEraser = !isEraser;
    eraserBtn.innerText = isEraser ? "橡皮擦：開啟" : "橡皮擦：關閉";
    updatePen();
}

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    digitDisplay.innerText = "---";
    confDetails.innerText = "星域已淨化，請重新書寫";
}

async function toggleCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        video.style.display = "none";
        camToggleBtn.innerText = "📷 開啟鏡頭";
        clearInterval(realtimeInterval);
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = cameraStream;
            video.style.display = "block";
            camToggleBtn.innerText = "📷 關閉鏡頭";
            realtimeInterval = setInterval(() => predict(), 800);
        } catch (err) { alert("鏡頭開啟失敗: " + err); }
    }
}

function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            clearCanvas();
            ctx.drawImage(img, 50, 50, canvas.width - 100, canvas.height - 100);
            predict();
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function initSpeechRecognition() {
    const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Speech) return;
    recognition = new Speech();
    recognition.lang = 'zh-TW';
    recognition.onresult = (e) => {
        if (e.results[0][0].transcript.includes('清除')) clearCanvas();
    };
}

function toggleVoice() {
    if (!recognition) return;
    if (isVoiceActive) recognition.stop(); else recognition.start();
    isVoiceActive = !isVoiceActive;
    voiceBtn.classList.toggle('active', isVoiceActive);
}

function updateDetails(data) {
    if (data.length === 0) return;
    let html = "<b>辨識詳細資訊：</b><br>";
    data.forEach((item, i) => {
        html += `數字 ${i+1}: <span style="color:#a3d9ff">${item.digit}</span> (${item.conf})<br>`;
    });
    confDetails.innerHTML = html;
}

// --- 5. 事件監聽 (滑鼠/觸控/按鈕) ---
function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: cx - rect.left, y: cy - rect.top };
}

canvas.addEventListener('mousedown', (e) => { isDrawing = true; ctx.beginPath(); const p = getPos(e); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('mousemove', (e) => { if (!isDrawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
canvas.addEventListener('mouseup', () => { isDrawing = false; predict(); });

// 觸控支援
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); isDrawing = true; ctx.beginPath(); const p = getPos(e); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if (!isDrawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
canvas.addEventListener('touchend', () => { isDrawing = false; predict(); });

// 按鈕事件綁定
// 注意：請確保 HTML 中按鈕的 onclick="predict()" 等標籤已移除，或直接在這裡綁定
document.querySelector('button[onclick="predict()"]').onclick = predict;
document.querySelector('button[onclick="clearCanvas()"]').onclick = clearCanvas;
document.getElementById('eraserBtn').onclick = toggleEraser;
document.getElementById('camToggleBtn').onclick = toggleCamera;
document.getElementById('voiceBtn').onclick = toggleVoice;
document.getElementById('fileInput').onchange = handleFile;

init();
