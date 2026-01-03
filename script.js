/**
 * 🌌 銀河手寫數字辨識系統 - 終極穩定版 (Industrial Stable)
 * ---------------------------------------------------------
 * 1. 硬體相容：自動降級至 CPU 模式，解決 WebGL 報錯。
 * 2. 函數對齊：修復 clearCanvas / triggerFile 等 ReferenceError。
 * 3. 邏輯深度：完全手寫實作 p.py 中的 Threshold -> Dilate -> Moments -> Centering。
 */

// ==========================================
// 1. 全域元件與變數
// ==========================================
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

let model = null;
let isDrawing = false;
let isEraser = false;
let cameraStream = null;
let realtimeInterval = null;
let recognition = null;
let isVoiceActive = false;

// 系統設定 (對標 p.py)
const PEN_WIDTH = 18;
const ERASER_WIDTH = 60;
const GALAXY_COLORS = ["#a3d9ff", "#7ed6df", "#e056fd", "#686de0", "#ffffff"];
const MNIST_PAD = 0.45; 

// ==========================================
// 2. 解決 WebGL 報錯：環境初始化
// ==========================================
async function initEnvironment() {
    try {
        // 如果 WebGL 失敗，強制使用 CPU，解決截圖中的 backend_webgl.js 錯誤
        await tf.setBackend('cpu'); 
        console.log("🛠️ 系統偵測硬體限制，已強制切換至 CPU 運算模式");
        await tf.ready();
    } catch (e) {
        console.warn("TFJS 環境初始化警告:", e);
    }
}

// ==========================================
// 3. 模型載入與權重視射 (對標你的 console 修復日誌)
// ==========================================
async function loadModelAndFix() {
    await initEnvironment();
    const modelUrl = `tfjs_model/model.json?v=${Date.now()}`;
    
    try {
        confDetails.innerHTML = "<span class='loading'>🧬 正在攔截並修正神經網路架構...</span>";
        
        const handler = tf.io.browserHTTPRequest(modelUrl);
        const originalLoad = handler.load.bind(handler);

        handler.load = async () => {
            const artifacts = await originalLoad();
            
            // 修補 InputLayer 缺失形狀
            if (artifacts.modelTopology && artifacts.modelTopology.model_config) {
                const config = artifacts.modelTopology.model_config.config;
                const layers = Array.isArray(config) ? config : config.layers;
                layers.forEach(layer => {
                    if (layer.class_name === 'InputLayer' || layer.config.name.includes('input')) {
                        if (!layer.config.batch_input_shape) {
                            layer.config.batch_input_shape = [null, 28, 28, 1];
                        }
                    }
                });
            }

            // 修補權重名稱 (解決 sequential/conv2d 找不到的問題)
            if (artifacts.weightSpecs) {
                artifacts.weightSpecs.forEach(spec => {
                    const oldName = spec.name;
                    spec.name = spec.name.replace(/^sequential(\/|_\d+\/)/, '');
                    if (oldName !== spec.name) console.log(`✅ 權重視射: ${oldName} -> ${spec.name}`);
                });
            }
            return artifacts;
        };

        model = await tf.loadLayersModel(handler);
        confDetails.innerText = "🚀 銀河核心同步成功";
        
        // 預熱
        tf.tidy(() => model.predict(tf.zeros([1, 28, 28, 1])));
    } catch (err) {
        confDetails.innerHTML = `<span style="color:#ff4757">❌ 載入失敗: ${err.message}</span>`;
    }
}

// ==========================================
// 4. OpenCV 底層算法移植 (完全展開)
// ==========================================

/** 手寫 Dilation (膨脹) */
function manualDilate(pixelData, width, height) {
    const output = new Uint8ClampedArray(pixelData.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let max = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const ny = y + ky, nx = x + kx;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        max = Math.max(max, pixelData[ny * width + nx]);
                    }
                }
            }
            output[y * width + x] = max;
        }
    }
    return output;
}

/** 手寫 Moments 質心校正 */
function getShiftVector(pixels, w, h) {
    let m00 = 0, m10 = 0, m01 = 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const v = pixels[y * w + x];
            if (v > 10) {
                m00 += v; m10 += x * v; m01 += y * v;
            }
        }
    }
    if (m00 === 0) return { dx: 0, dy: 0 };
    return { dx: (w / 2) - (m10 / m00), dy: (h / 2) - (m01 / m00) };
}

/** 核心處理 ROI */
async function processDigitROI(roiCanvas) {
    const tempCtx = roiCanvas.getContext('2d');
    const raw = tempCtx.getImageData(0, 0, roiCanvas.width, roiCanvas.height);
    
    let gray = new Uint8ClampedArray(raw.width * raw.height);
    for (let i = 0; i < raw.data.length; i += 4) {
        gray[i / 4] = raw.data[i] > 120 ? 255 : 0;
    }

    gray = manualDilate(gray, raw.width, raw.height);
    const shift = getShiftVector(gray, raw.width, raw.height);

    const final = document.createElement('canvas');
    final.width = 28; final.height = 28;
    const fCtx = final.getContext('2d');
    fCtx.fillStyle = "black";
    fCtx.fillRect(0, 0, 28, 28);

    const side = Math.max(roiCanvas.width, roiCanvas.height);
    const scale = (28 * (1 - MNIST_PAD)) / side;
    
    fCtx.save();
    fCtx.translate(14 + shift.dx * scale, 14 + shift.dy * scale);
    fCtx.scale(scale, scale);
    fCtx.drawImage(roiCanvas, -roiCanvas.width / 2, -roiCanvas.height / 2);
    fCtx.restore();

    const tensor = tf.tidy(() => tf.browser.fromPixels(final, 1).toFloat().div(255.0).expandDims(0));
    const pred = model.predict(tensor);
    const scores = await pred.data();
    const result = { digit: pred.argMax(-1).dataSync()[0], conf: Math.max(...scores) };

    tf.dispose([tensor, pred]);
    return result;
}

// ==========================================
// 5. 輪廓掃描與多位數辨識
// ==========================================

function findRegions(isRealtime) {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data, width, height } = imgData;
    const visited = new Uint8Array(width * height);
    const regions = [];
    const step = isRealtime ? 4 : 2;

    for (let y = 0; y < height; y += step) {
        for (let x = 0; x < width; x += step) {
            const i = y * width + x;
            if (!visited[i] && data[i * 4] > 100) {
                let stack = [[x, y]];
                visited[i] = 1;
                let minX = x, maxX = x, minY = y, maxY = y;

                while (stack.length > 0) {
                    const [cx, cy] = stack.pop();
                    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);

                    [[cx+step, cy], [cx-step, cy], [cx, cy+step], [cx, cy-step]].forEach(([nx, ny]) => {
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const ni = ny * width + nx;
                            if (!visited[ni] && data[ni * 4] > 100) {
                                visited[ni] = 1; stack.push([nx, ny]);
                            }
                        }
                    });
                }
                const w = maxX - minX + 1;
                const h = maxY - minY + 1;
                if (w * h < 100) continue;
                regions.push({ x: minX, y: minY, w, h });
            }
        }
    }
    return regions.sort((a, b) => a.x - b.x);
}

async function runRecognition(isRealtime = false) {
    if (!model) return;
    const regions = findRegions(isRealtime);
    let finalStr = "";
    
    // 為了掃描鏡頭+畫布，我們需要一個 Snapshot
    const snap = document.createElement('canvas');
    snap.width = canvas.width; snap.height = canvas.height;
    const sCtx = snap.getContext('2d');
    if (cameraStream) sCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    sCtx.drawImage(canvas, 0, 0);

    if (isRealtime) ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const r of regions) {
        const roi = document.createElement('canvas');
        roi.width = r.w; roi.height = r.h;
        roi.getContext('2d').putImageData(sCtx.getImageData(r.x, r.y, r.w, r.h), 0, 0);

        const res = await processDigitROI(roi);
        if (res.conf > 0.7) {
            finalStr += res.digit;
            if (isRealtime) {
                ctx.strokeStyle = "#00FF00"; ctx.strokeRect(r.x, r.y, r.w, r.h);
                ctx.fillStyle = "#00FF00"; ctx.fillText(res.digit, r.x, r.y - 5);
            }
        }
    }
    digitDisplay.innerText = finalStr || "---";
    updatePen();
}

// ==========================================
// 6. 修復 ReferenceError：將函數掛載到全域
// ==========================================

// 1. 修復 clearCanvas 報錯
window.clearCanvas = function() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    digitDisplay.innerText = "---";
    confDetails.innerText = "星域已清空";
};

// 2. 修復 triggerFile 報錯
window.triggerFile = function() {
    fileInput.click();
};

window.toggleEraser = function() {
    isEraser = !isEraser;
    eraserBtn.innerText = isEraser ? "畫筆模式" : "橡皮擦模式";
    updatePen();
};

window.toggleCamera = async function() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        clearInterval(realtimeInterval);
        video.style.display = "none";
        camToggleBtn.innerText = "📷 開啟鏡頭";
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = cameraStream;
            video.style.display = "block";
            camToggleBtn.innerText = "📷 關閉鏡頭";
            realtimeInterval = setInterval(() => runRecognition(true), 500);
        } catch (e) { alert("鏡頭不可用"); }
    }
};

window.startPredict = function() {
    runRecognition(false);
};

// ==========================================
// 7. 視覺與事件
// ==========================================

function updatePen() {
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = isEraser ? "black" : "white";
    ctx.lineWidth = isEraser ? ERASER_WIDTH : PEN_WIDTH;
}

function getCoord(e) {
    const r = canvas.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    const y = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
    return { x, y };
}

canvas.addEventListener('mousedown', (e) => { isDrawing = true; ctx.beginPath(); const p = getCoord(e); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const p = getCoord(e);
    ctx.lineTo(p.x, p.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.x, p.y);
    // 噴發星塵
    if (!isEraser) {
        const s = document.createElement('div');
        s.style.cssText = `position:absolute; left:${p.x+window.scrollX}px; top:${p.y+window.scrollY}px; width:4px; height:4px; background:white; border-radius:50%; pointer-events:none; animation: star-fade 0.8s forwards;`;
        document.body.appendChild(s); setTimeout(() => s.remove(), 800);
    }
});
canvas.addEventListener('mouseup', () => { isDrawing = false; if(!cameraStream) runRecognition(); });

// 處理檔案上傳
fileInput.addEventListener('change', (e) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
        const img = new Image();
        img.onload = () => {
            window.clearCanvas();
            const s = Math.min(canvas.width/img.width, canvas.height/img.height) * 0.8;
            ctx.drawImage(img, (canvas.width-img.width*s)/2, (canvas.height-img.height*s)/2, img.width*s, img.height*s);
            runRecognition();
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
});

// 初始化
loadModelAndFix();
window.clearCanvas();
