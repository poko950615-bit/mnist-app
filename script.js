/**
 * 🌌 銀河手寫數字辨識系統 - 終極全功能整合版 (No-省略版)
 * 整合：模型修復、Python 邏輯 JS 化、銀河特效、語音控制
 */

// --- 1. 常數與全域變數 ---
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

let model = null;
let isDrawing = false;
let isEraser = false;
let cameraStream = null;
let realtimeInterval = null;
let recognition = null;
let isVoiceActive = false;
let lastX = 0;
let lastY = 0;

// --- 🛠️ 核心修復載入器 (解決 Keras v3 / TFJS 相容性) ---
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

        // 修復 B: 移除權重名稱中的前綴
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
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();
    addGalaxyEffects();

    const modelUrl = `tfjs_model/model.json?t=${Date.now()}`;
    try {
        confDetails.innerText = "🌌 正在啟動銀河 AI 引擎...";
        await tf.setBackend('cpu'); // 使用 CPU 確保相容性
        await tf.ready();
        model = await tf.loadLayersModel(new PatchModelLoader(modelUrl));
        console.log("✅ 模型載入成功！");
        confDetails.innerText = "🚀 系統就緒，請開始在星域書寫";
        tf.tidy(() => model.predict(tf.zeros([1, 28, 28, 1])));
    } catch (err) {
        console.error("初始化失敗:", err);
        confDetails.innerHTML = `<span style="color:red">❌ 載入失敗: ${err.message}</span>`;
    }
}

// --- 3. 影像處理邏輯 (原 p.py advanced_preprocess 的 JS 實作) ---
function advancedPreprocessJS(roiTensor) {
    return tf.tidy(() => {
        // 1. 強化筆畫 (膨脹效果模擬)
        const kernel = tf.ones([2, 2, 1, 1]);
        let dilated = tf.dilation2d(roiTensor.expandDims(0).expandDims(-1), kernel, [1, 1, 1, 1], 'same').squeeze();

        // 2. 動態 Padding
        const [h, w] = dilated.shape;
        const padSize = Math.floor(Math.max(h, w) * 0.45);
        const padded = dilated.pad([[padSize, padSize], [padSize, padSize]], 0);

        // 3. 縮放至 28x28
        let resized = tf.image.resizeBilinear(padded.expandDims(-1), [28, 28]);

        // 4. 質心校正 (Centroid alignment)
        const sum = resized.sum();
        if (sum.dataSync()[0] !== 0) {
            const indices = tf.meshgrid(tf.range(0, 28), tf.range(0, 28));
            const xCenter = resized.mul(indices[0].expandDims(-1)).sum().div(sum).dataSync()[0];
            const yCenter = resized.mul(indices[1].expandDims(-1)).sum().div(sum).dataSync()[0];
            
            // 模擬 cv2.warpAffine 的平移
            resized = tf.image.transform(resized.expandDims(0), [1, 0, xCenter - 14, 0, 1, yCenter - 14, 0, 0]).squeeze(0);
        }

        return resized.div(255.0).expandDims(0);
    });
}

// 實作 OpenCV 的連通區域分析 (JS 版)
function findComponents(pixels, width, height, isRealtime) {
    const visited = new Uint8Array(width * height);
    const components = [];
    const MIN_AREA = isRealtime ? 500 : 150;

    for (let y = 0; y < height; y += 4) {
        for (let x = 0; x < width; x += 4) {
            const idx = y * width + x;
            if (!visited[idx] && pixels[idx * 4] > 100) {
                let q = [[x, y]];
                visited[idx] = 1;
                let minX = x, maxX = x, minY = y, maxY = y, area = 0;
                
                while (q.length > 0) {
                    const [cx, cy] = q.shift();
                    area++;
                    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);

                    [[cx+4, cy], [cx-4, cy], [cx, cy+4], [cx, cy-4]].forEach(([nx, ny]) => {
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            if (!visited[nIdx] && pixels[nIdx * 4] > 100) {
                                visited[nIdx] = 1; q.push([nx, ny]);
                            }
                        }
                    });
                }

                const w = maxX - minX + 1;
                const h = maxY - minY + 1;
                const aspectRatio = w / h;
                const solidity = area / (w * h / 16); // 估算值

                // Python 端的強力過濾邏輯
                if (area * 16 < MIN_AREA) continue;
                if (aspectRatio > 2.5 || aspectRatio < 0.15) continue;
                if (solidity < 0.15) continue;

                components.push({ x: minX, y: minY, w, h, area: area * 16 });
            }
        }
    }
    return components.sort((a, b) => a.x - b.x);
}

// --- 4. 辨識主程序 ---
async function predict(isRealtime = false) {
    if (!model) return;

    // 模擬 Python 端的畫面抓取
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width; tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext('2d');
    if (cameraStream) tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    tCtx.drawImage(canvas, 0, 0);

    const imgData = tCtx.getImageData(0, 0, canvas.width, canvas.height);
    const comps = findComponents(imgData.data, canvas.width, canvas.height, isRealtime);
    
    let finalRes = "";
    let details = [];
    let validBoxes = [];

    if (isRealtime) ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const comp of comps) {
        const { x, y, w, h } = comp;
        
        // 建立區域畫布
        const roiCanvas = document.createElement('canvas');
        roiCanvas.width = w; roiCanvas.height = h;
        roiCanvas.getContext('2d').putImageData(tCtx.getImageData(x, y, w, h), 0, 0);

        // 連體字切割 (Python 邏輯)
        if (w > h * 1.3) {
            const splitX = Math.floor(w / 2);
            const subWidths = [splitX, w - splitX];
            const subOffsets = [0, splitX];

            for (let i = 0; i < 2; i++) {
                const subCanvas = document.createElement('canvas');
                subCanvas.width = subWidths[i]; subCanvas.height = h;
                subCanvas.getContext('2d').drawImage(roiCanvas, subOffsets[i], 0, subWidths[i], h, 0, 0, subWidths[i], h);
                const result = await runModel(subCanvas);
                if (result.conf > 0.8) {
                    finalRes += result.digit;
                    details.push(result);
                }
            }
            continue;
        }

        const result = await runModel(roiCanvas);
        if (isRealtime && result.conf < 0.85) continue;

        if (result.conf > 0.7) {
            finalRes += result.digit;
            details.push(result);
            validBoxes.push(comp);
            
            if (isRealtime) {
                ctx.strokeStyle = "#00FF00"; ctx.lineWidth = 3;
                ctx.strokeRect(x, y, w, h);
                ctx.fillStyle = "#00FF00"; ctx.font = "bold 24px Arial";
                ctx.fillText(result.digit, x, y - 5);
            }
        }
    }

    digitDisplay.innerText = finalRes || "---";
    updateDetails({ details });
    if (isRealtime) updatePen();
}

async function runModel(roiCanvas) {
    const tensor = tf.browser.fromPixels(roiCanvas, 1).toFloat();
    const processed = advancedPreprocessJS(tensor);
    const pred = model.predict(processed);
    const data = await pred.data();
    const digit = pred.argMax(-1).dataSync()[0];
    const conf = Math.max(...data);
    
    tf.dispose([tensor, processed, pred]);
    return { digit, conf: (conf * 100).toFixed(1) + "%" };
}

// --- 5. UI 與特效功能 ---
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
    if (isEraser) addVisualFeedback("#e74c3c");
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!cameraStream) { ctx.fillStyle = "black"; ctx.fillRect(0, 0, canvas.width, canvas.height); }
    digitDisplay.innerText = "---";
    confDetails.innerText = "畫布已清空，銀河已淨空";
    addVisualFeedback("#2ecc71");
    addGalaxyEffects();
}

function addVisualFeedback(color) {
    const btns = document.querySelectorAll('button');
    btns.forEach(btn => {
        btn.style.boxShadow = `0 0 20px ${color}`;
        setTimeout(() => { btn.style.boxShadow = ""; }, 300);
    });
}

async function toggleCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        clearInterval(realtimeInterval);
        video.style.display = "none";
        mainBox.classList.remove('cam-active');
        camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 開啟鏡頭';
        init();
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 1280, height: 720 } });
            video.srcObject = cameraStream;
            video.style.display = "block";
            mainBox.classList.add('cam-active');
            camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 關閉鏡頭';
            realtimeInterval = setInterval(() => predict(true), 400);
            clearCanvas();
        } catch (err) { alert("鏡頭失敗: " + err); }
    }
}

// --- 6. 事件監聽 (繪圖與觸控) ---
function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: cx - rect.left, y: cy - rect.top };
}

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true; const {x, y} = getCanvasPos(e);
    ctx.beginPath(); ctx.moveTo(x, y);
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const {x, y} = getCanvasPos(e);
    ctx.lineTo(x, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y);
    if (!isEraser) addDrawingEffect(x, y);
});

function stopDrawing() {
    if (isDrawing) { isDrawing = false; if (!cameraStream) predict(); }
}
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); if (e.touches.length === 1) { isDrawing = true; const {x, y} = getCanvasPos(e); ctx.beginPath(); ctx.moveTo(x, y); } });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if (isDrawing) { const {x, y} = getCanvasPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y); } });
canvas.addEventListener('touchend', stopDrawing);

function addDrawingEffect(x, y) {
    const effect = document.createElement('div');
    effect.className = "drawing-dot";
    effect.style.left = x + 'px'; effect.style.top = y + 'px';
    document.body.appendChild(effect);
    setTimeout(() => effect.remove(), 500);
}

// --- 7. 語音與檔案功能 ---
function initSpeechRecognition() {
    const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Speech) return;
    recognition = new Speech();
    recognition.lang = 'zh-TW';
    recognition.onresult = (e) => {
        const text = e.results[e.results.length - 1][0].transcript;
        if (text.includes('清除')) clearCanvas();
        else if (text.includes('辨識')) predict();
    };
}

function toggleVoice() {
    if (!recognition) return;
    if (isVoiceActive) recognition.stop(); else recognition.start();
    isVoiceActive = !isVoiceActive;
    voiceBtn.classList.toggle('voice-active', isVoiceActive);
}

function handleFile(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            clearCanvas();
            ctx.drawImage(img, 50, 50, canvas.width - 100, canvas.height - 100);
            predict();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function updateDetails(data) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (!data.details || data.details.length === 0) html += "等待中...";
    else data.details.forEach((item, i) => {
        html += `數字 ${i + 1}: <b style="color:#a3d9ff">${item.digit}</b> (${item.conf})<br>`;
    });
    confDetails.innerHTML = html;
}

// 綁定 HTML 按鈕 (假設原本 onclick 已移除)
camToggleBtn.onclick = toggleCamera;
eraserBtn.onclick = toggleEraser;
voiceBtn.onclick = toggleVoice;
fileInput.onchange = handleFile;
document.querySelector('button[onclick="clearCanvas()"]').onclick = clearCanvas;
document.querySelector('button[onclick="predict()"]').onclick = () => predict();
document.querySelector('button[onclick="triggerFile()"]').onclick = () => fileInput.click();

init();
