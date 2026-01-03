/**
 * 🌌 銀河手寫數字辨識系統 - 終極全功能整合版
 * 整合：模型修復、語音控制、檔案上傳、銀河特效、即時鏡頭、進階分割
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

// --- 🛠️ 超級修復載入器 (Super Patch Loader) ---
// 針對 TensorFlow.js 與 Keras 3.x 轉換後的相容性修補
class PatchModelLoader {
    constructor(url) { this.url = url; }
    
    async load() {
        const loader = tf.io.browserHTTPRequest(this.url);
        const artifacts = await loader.load();
        
        console.log("🛠️ 正在執行深度修復...");

        // 修復 A: 注入 InputLayer 形狀
        const traverseAndPatch = (obj) => {
            if (!obj || typeof obj !== 'object') return;
            if (obj.class_name === 'InputLayer' && obj.config) {
                const cfg = obj.config;
                if (!cfg.batchInputShape && !cfg.batch_input_shape) {
                    console.log(`🔧 [修復 A] 注入形狀 [null, 28, 28, 1]`);
                    cfg.batchInputShape = [null, 28, 28, 1];
                }
            }
            if (Array.isArray(obj)) obj.forEach(item => traverseAndPatch(item));
            else Object.keys(obj).forEach(key => traverseAndPatch(obj[key]));
        };
        if (artifacts.modelTopology) traverseAndPatch(artifacts.modelTopology);

        // 修復 B: 修正權重名稱 (移除 sequential/ 前綴)
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
        await tf.setBackend('cpu');
        await tf.ready();

        model = await tf.loadLayersModel(new PatchModelLoader(modelUrl));
        
        console.log("✅ 系統全線就緒！");
        confDetails.innerText = "🚀 系統就緒，請開始在星域書寫";
        
        // 暖身預測
        tf.tidy(() => model.predict(tf.zeros([1, 28, 28, 1])));
    } catch (err) {
        console.error("初始化失敗:", err);
        confDetails.innerHTML = `<span style="color:red">❌ 載入失敗: ${err.message}</span>`;
    }
}

// --- 3. 核心辨識與影像處理 ---
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
        roiCanvas.width = w; roiCanvas.height = h;
        const roiCtx = roiCanvas.getContext('2d');
        roiCtx.drawImage(canvas, x, y, w, h, 0, 0, w, h);

        // --- 進階邏輯：分割寬度過大的連字 (如 11) ---
        if (w > h * 1.3) {
            const splitX = Math.floor(w / 2);
            const widths = [splitX, w - splitX];
            const offsets = [0, splitX];

            for (let i = 0; i < 2; i++) {
                const subCanvas = document.createElement('canvas');
                subCanvas.width = widths[i]; subCanvas.height = h;
                subCanvas.getContext('2d').drawImage(roiCanvas, offsets[i], 0, widths[i], h, 0, 0, widths[i], h);
                
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

        // 一般辨識
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
        // 在鏡頭模式下繪製綠色偵測框
        validBoxes.forEach((box, i) => {
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 3;
            ctx.strokeRect(box.x, box.y, box.w, box.h);
        });
    }
}

// 連通域算法：找尋獨立數字
function findDigitBoxes(imageData) {
    const { data, width, height } = imageData;
    const visited = new Uint8Array(width * height);
    const boxes = [];

    for (let y = 0; y < height; y += 5) {
        for (let x = 0; x < width; x += 5) {
            const idx = y * width + x;
            if (!visited[idx] && data[idx * 4] > 100) {
                let minX = x, maxX = x, minY = y, maxY = y, count = 0;
                let queue = [[x, y]];
                visited[idx] = 1;

                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    count++;
                    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);
                    [[cx+10, cy], [cx-10, cy], [cx, cy+10], [cx, cy-10]].forEach(([nx, ny]) => {
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            if (!visited[nIdx] && data[nIdx * 4] > 100) {
                                visited[nIdx] = 1; queue.push([nx, ny]);
                            }
                        }
                    });
                }
                boxes.push({ x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1, area: count * 25 });
            }
        }
    }
    return boxes.sort((a, b) => a.x - b.x);
}

// --- 4. UI 視覺與互動功能 ---
function addGalaxyEffects() {
    // 增加一些點綴星光
    ctx.fillStyle = "rgba(163, 217, 255, 0.3)";
    ctx.beginPath(); ctx.arc(600, 50, 2, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(50, 350, 1.5, 0, Math.PI * 2); ctx.fill();
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
    ctx.fillStyle = "black"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    digitDisplay.innerText = "---";
    addGalaxyEffects();
}

async function toggleCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        video.style.display = "none";
        camToggleBtn.innerText = "📷 開啟鏡頭";
        clearInterval(realtimeInterval);
        init(); 
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = cameraStream;
            video.style.display = "block";
            camToggleBtn.innerText = "📷 關閉鏡頭";
            realtimeInterval = setInterval(() => predict(), 500);
        } catch (err) { alert("鏡頭故障: " + err); }
    }
}

// 語音辨識初始化
function initSpeechRecognition() {
    const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Speech) return;
    recognition = new Speech();
    recognition.lang = 'zh-TW';
    recognition.onresult = (e) => {
        const text = e.results[0][0].transcript;
        if (text.includes('清除')) clearCanvas();
    };
}

function toggleVoice() {
    if (!recognition) return;
    if (isVoiceActive) recognition.stop(); else recognition.start();
    isVoiceActive = !isVoiceActive;
    voiceBtn.classList.toggle('voice-active', isVoiceActive);
}

// 檔案上傳
function triggerFile() { fileInput.click(); }
function handleFile(e) {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            clearCanvas();
            ctx.drawImage(img, 50, 50, canvas.width-100, canvas.height-100);
            predict();
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function updateDetails(data) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (data.length === 0) html += "等待輸入...";
    else data.forEach((item, i) => {
        html += `數字 ${i+1}: <b style="color:#a3d9ff">${item.digit}</b> (${item.conf})<br>`;
    });
    confDetails.innerHTML = html;
}

// --- 5. 事件監聽 ---
function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    return { x, y };
}

canvas.addEventListener('mousedown', (e) => { isDrawing = true; const p = getPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('mousemove', (e) => { if (!isDrawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
canvas.addEventListener('mouseup', () => { isDrawing = false; predict(); });
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); isDrawing = true; const p = getPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if (!isDrawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
canvas.addEventListener('touchend', () => { isDrawing = false; predict(); });

init();
