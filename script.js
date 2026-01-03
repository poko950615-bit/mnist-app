/**
 * 🌌 銀河手寫數字辨識系統 - 究極整合版 (No Omissions)
 * 整合：
 * 1. Python p.py 全套影像增強 (膨脹、質心校正、連體字切割)
 * 2. TensorFlow.js 模型自動修復載入器
 * 3. 銀河視覺特效與語音控制系統
 */

// --- [1. 變數宣告與元素定義] ---
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

// --- [2. 模型修復載入器] ---
// 解決 Keras v3 轉換至 TFJS 時的結構缺失與權重命名錯誤
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

        // 修復 B: 移除 'sequential/' 命名衝突
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

// --- [3. 系統初始化] ---
async function init() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();
    addGalaxyEffects();

    const modelUrl = `tfjs_model/model.json?t=${Date.now()}`;
    try {
        confDetails.innerText = "🌌 正在同步銀河 AI 引擎...";
        await tf.ready();
        // 如果瀏覽器效能較弱，可改為 'cpu'，否則預設使用 'webgl'
        model = await tf.loadLayersModel(new PatchModelLoader(modelUrl));
        console.log("✅ 引擎啟動成功");
        confDetails.innerText = "🚀 系統就緒，星域等待書寫";
        // 暖身
        tf.tidy(() => model.predict(tf.zeros([1, 28, 28, 1])));
    } catch (err) {
        console.error("載入失敗:", err);
        confDetails.innerHTML = `<span style="color:red">❌ 引擎崩潰: ${err.message}</span>`;
    }
}

// --- [4. 影像處理核心 - Python 邏輯 JS 移植] ---

/**
 * 模擬 Python advanced_preprocess
 * 包含膨脹、動態 Padding、質心校正
 */
function advancedPreprocessJS(roiTensor) {
    return tf.tidy(() => {
        // 1. 轉為灰階並標準化
        let tensor = roiTensor.toFloat();
        
        // 2. 筆畫強化 (膨脹) - 使用 MaxPool 模擬 Dilation
        tensor = tensor.expandDims(0).expandDims(-1);
        tensor = tf.maxPool(tensor, [2, 2], [1, 1], 'same');
        tensor = tensor.squeeze();

        // 3. 動態 Padding (45% 比例)
        const [h, w] = tensor.shape;
        const padSize = Math.floor(Math.max(h, w) * 0.45);
        const padded = tensor.pad([[padSize, padSize], [padSize, padSize]], 0);

        // 4. 縮放至 28x28
        let resized = tf.image.resizeBilinear(padded.expandDims(-1), [28, 28]);

        // 5. 質心校正 (Centroid alignment)
        const moments = resized.sum();
        if (moments.dataSync()[0] > 0) {
            const rowSum = resized.sum(1).squeeze();
            const colSum = resized.sum(0).squeeze();
            const rows = tf.range(0, 28);
            const cols = tf.range(0, 28);
            
            const cy = rowSum.mul(rows).sum().div(moments).dataSync()[0];
            const cx = colSum.mul(cols).sum().div(moments).dataSync()[0];
            
            // 計算偏移量並應用平移
            const tx = 14 - cx;
            const ty = 14 - cy;
            
            // 使用 tf.image.transform 進行平移
            resized = tf.image.transform(
                resized.expandDims(0),
                [1, 0, -tx, 0, 1, -ty, 0, 0],
                'bilinear'
            ).squeeze(0);
        }

        // 最後標準化到 0-1
        return resized.div(255.0).expandDims(0);
    });
}

/**
 * 連通區域偵測 (取代 Python cv2.connectedComponentsWithStats)
 */
function findDigitBoxes(pixels, width, height, isRealtime) {
    const visited = new Uint8Array(width * height);
    const boxes = [];
    const MIN_AREA = isRealtime ? 500 : 150;

    for (let y = 0; y < height; y += 4) {
        for (let x = 0; x < width; x += 4) {
            const idx = y * width + x;
            // 偵測白色筆畫 (R值 > 100)
            if (!visited[idx] && pixels[idx * 4] > 100) {
                let queue = [[x, y]];
                visited[idx] = 1;
                let minX = x, maxX = x, minY = y, maxY = y, count = 0;

                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    count++;
                    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);

                    // 檢查鄰近像素 (步長需與掃描步長一致)
                    [[cx+4, cy], [cx-4, cy], [cx, cy+4], [cx, cy-4]].forEach(([nx, ny]) => {
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            if (!visited[nIdx] && pixels[nIdx * 4] > 100) {
                                visited[nIdx] = 1;
                                queue.push([nx, ny]);
                            }
                        }
                    });
                }

                const w = maxX - minX + 1;
                const h = maxY - minY + 1;
                const area = count * 16;
                const aspectRatio = w / h;

                // 強力過濾邏輯 (與 p.py 同步)
                if (area < MIN_AREA) continue;
                if (aspectRatio > 2.5 || aspectRatio < 0.15) continue;
                if (area / (w * h) < 0.1) continue; // Solidity

                boxes.push({ x: minX, y: minY, w, h, area });
            }
        }
    }
    return boxes.sort((a, b) => a.x - b.x); // 從左到右排序
}

// --- [5. 辨識與執行程序] ---

async function runPrediction(roiCanvas) {
    const tensor = tf.browser.fromPixels(roiCanvas, 1);
    const processed = advancedPreprocessJS(tensor);
    const prediction = model.predict(processed);
    const scores = await prediction.data();
    const digit = prediction.argMax(-1).dataSync()[0];
    const confidence = Math.max(...scores);

    tf.dispose([tensor, processed, prediction]);
    return { digit, conf: confidence };
}

async function predict(isRealtime = false) {
    if (!model) return;

    // 建立快照
    const snapshotCanvas = document.createElement('canvas');
    snapshotCanvas.width = canvas.width;
    snapshotCanvas.height = canvas.height;
    const sCtx = snapshotCanvas.getContext('2d');
    if (cameraStream) sCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    sCtx.drawImage(canvas, 0, 0);

    const imgData = sCtx.getImageData(0, 0, canvas.width, canvas.height);
    const boxes = findDigitBoxes(imgData.data, canvas.width, canvas.height, isRealtime);
    
    let finalDigits = "";
    let detailsList = [];

    // 若是即時模式，先清空畫布上的舊綠框
    if (isRealtime) ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const box of boxes) {
        const roiCanvas = document.createElement('canvas');
        roiCanvas.width = box.w; roiCanvas.height = box.h;
        roiCanvas.getContext('2d').putImageData(sCtx.getImageData(box.x, box.y, box.w, box.h), 0, 0);

        // 連體字分割邏輯 (p.py 1.3 門檻)
        if (box.w > box.h * 1.3) {
            const mid = Math.floor(box.w / 2);
            const subWidths = [mid, box.w - mid];
            const subOffsets = [0, mid];

            for (let i = 0; i < 2; i++) {
                const subCanvas = document.createElement('canvas');
                subCanvas.width = subWidths[i]; subCanvas.height = box.h;
                subCanvas.getContext('2d').drawImage(roiCanvas, subOffsets[i], 0, subWidths[i], box.h, 0, 0, subWidths[i], box.h);
                
                const res = await runPrediction(subCanvas);
                if (res.conf > 0.8) {
                    finalDigits += res.digit;
                    detailsList.push({ digit: res.digit, conf: (res.conf * 100).toFixed(1) + "%" });
                }
            }
        } else {
            // 一般辨識
            const res = await runPrediction(roiCanvas);
            // 即時模式信心度門檻 0.85
            if (isRealtime && res.conf < 0.85) continue;

            if (res.conf > 0.7) {
                finalDigits += res.digit;
                detailsList.push({ digit: res.digit, conf: (res.conf * 100).toFixed(1) + "%" });

                if (isRealtime) {
                    // 畫出偵測框與結果
                    ctx.strokeStyle = "#00FF00";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(box.x, box.y, box.w, box.h);
                    ctx.fillStyle = "#00FF00";
                    ctx.font = "bold 24px Arial";
                    ctx.fillText(res.digit, box.x, box.y - 5);
                }
            }
        }
    }

    digitDisplay.innerText = finalDigits || "---";
    updateDetailsDisplay(detailsList);
    if (isRealtime) updatePen(); // 恢復畫筆設定
}

// --- [6. UI 互動與視覺效果] ---

function addGalaxyEffects() {
    setTimeout(() => {
        if (!cameraStream) {
            ctx.fillStyle = "rgba(163, 217, 255, 0.2)";
            ctx.beginPath(); ctx.arc(600, 40, 2, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(50, 320, 1.5, 0, Math.PI * 2); ctx.fill();
            updatePen();
        }
    }, 500);
}

function updatePen() {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = isEraser ? "black" : "white";
    ctx.lineWidth = isEraser ? 40 : 15;
}

function toggleEraser() {
    isEraser = !isEraser;
    eraserBtn.innerText = isEraser ? "橡皮擦：開啟" : "橡皮擦：關閉";
    eraserBtn.classList.toggle('eraser-active', isEraser);
    updatePen();
    addVisualFeedback(isEraser ? "#e74c3c" : "#3498db");
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!cameraStream) {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    digitDisplay.innerText = "---";
    confDetails.innerText = "星域已清空，銀河已淨化";
    addVisualFeedback("#2ecc71");
    addGalaxyEffects();
}

function addVisualFeedback(color) {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
        const original = btn.style.boxShadow;
        btn.style.boxShadow = `0 0 15px ${color}`;
        setTimeout(() => btn.style.boxShadow = original, 300);
    });
}

async function toggleCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        if (realtimeInterval) clearInterval(realtimeInterval);
        video.style.display = "none";
        mainBox.classList.remove('cam-active');
        camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 開啟鏡頭';
        init();
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment", width: 1280, height: 720 }
            });
            video.srcObject = cameraStream;
            video.style.display = "block";
            mainBox.classList.add('cam-active');
            camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 關閉鏡頭';
            realtimeInterval = setInterval(() => predict(true), 400);
            clearCanvas();
        } catch (err) { alert("鏡頭故障: " + err); }
    }
}

// --- [7. 事件監聽] ---

function getXY(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: cx - rect.left, y: cy - rect.top };
}

function startDraw(e) {
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getXY(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function draw(e) {
    e.preventDefault();
    if (!isDrawing) return;
    const { x, y } = getXY(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    if (!isEraser) addStarParticle(x, y);
}

function stopDraw() {
    if (isDrawing) {
        isDrawing = false;
        if (!cameraStream) setTimeout(() => predict(false), 100);
    }
}

function addStarParticle(x, y) {
    const star = document.createElement('div');
    star.className = "drawing-dot"; // 需對應 CSS 樣式
    star.style.left = x + 'px';
    star.style.top = y + 'px';
    document.body.appendChild(star);
    setTimeout(() => star.remove(), 600);
}

// 綁定事件
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('touchstart', startDraw);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDraw);

// 語音識別
function initSpeechRecognition() {
    const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Speech) return;
    recognition = new Speech();
    recognition.lang = 'zh-TW';
    recognition.onresult = (e) => {
        const cmd = e.results[e.results.length - 1][0].transcript;
        if (cmd.includes('清除')) clearCanvas();
        if (cmd.includes('橡皮擦')) toggleEraser();
    };
}

function toggleVoice() {
    if (!recognition) return;
    if (isVoiceActive) recognition.stop(); else recognition.start();
    isVoiceActive = !isVoiceActive;
    voiceBtn.classList.toggle('voice-active', isVoiceActive);
}

// 檔案處理
function triggerFile() { fileInput.click(); }
function handleFile(e) {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            clearCanvas();
            const scale = Math.min(canvas.width/img.width, canvas.height/img.height) * 0.8;
            const w = img.width * scale;
            const h = img.height * scale;
            ctx.drawImage(img, (canvas.width-w)/2, (canvas.height-h)/2, w, h);
            predict(false);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function updateDetailsDisplay(details) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (details.length === 0) html += "等待數據...";
    else {
        details.forEach((item, i) => {
            html += `數字 ${i+1}: <span style="color:#a3d9ff">${item.digit}</span> (信心度: ${item.conf})<br>`;
        });
    }
    confDetails.innerHTML = html;
}

// 啟動
init();
