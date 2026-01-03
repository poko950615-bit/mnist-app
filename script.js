/**
 * 🌠 銀河手寫數字辨識系統 - 深度架構修復版
 * ---------------------------------------------------------
 * 針對 image_794f42.png 中的 "Provided weight data has no target variable" 進行修復
 * 1. 強制修復 Sequential 命名空間
 * 2. 手動注入 InputLayer Shape
 * 3. 完整移植 p.py 的 cv2.dilate 與 cv2.moments
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

// 銀河視覺與運算參數 (對標 p.py)
const PEN_WIDTH = 18;
const ERASER_WIDTH = 60;
const GALAXY_COLORS = ["#a3d9ff", "#7ed6df", "#e056fd", "#686de0", "#ffffff"];
const MNIST_PAD = 0.45; // 依照 p.py 設定 45% 邊界填充

// ==========================================
// 2. 核心：模型載入器與權重命名修復 (解決截圖中的報錯)
// ==========================================

async function loadModelAndFix() {
    const modelUrl = `tfjs_model/model.json?nocache=${Date.now()}`;
    
    try {
        confDetails.innerHTML = "<span class='loading'>🧬 正在攔截並修正神經網路架構...</span>";
        await tf.ready();

        // 建立自定義載入處理器，手動修改 JSON 內容
        const handler = tf.io.browserHTTPRequest(modelUrl);
        const originalLoad = handler.load.bind(handler);

        handler.load = async () => {
            const artifacts = await originalLoad();
            
            console.log("🛠️ 原始權重清單:", artifacts.weightSpecs.map(s => s.name));

            // [修復 1] 解決 "An InputLayer should be passed an inputShape" 錯誤
            if (artifacts.modelTopology && artifacts.modelTopology.model_config) {
                const config = artifacts.modelTopology.model_config.config;
                const layers = Array.isArray(config) ? config : config.layers;
                
                layers.forEach(layer => {
                    if (layer.class_name === 'InputLayer' || layer.config.name === 'conv2d_input') {
                        if (!layer.config.batch_input_shape) {
                            layer.config.batch_input_shape = [null, 28, 28, 1];
                        }
                    }
                });
            }

            // [修復 2] 解決 "weight data has no target variable" 錯誤
            // 截圖顯示報錯尋找 sequential/conv2d/kernel，所以我們必須移除權重清單中的 sequential 前綴
            if (artifacts.weightSpecs) {
                artifacts.weightSpecs.forEach(spec => {
                    // 將 "sequential/conv2d/kernel" 轉為 "conv2d/kernel"
                    const oldName = spec.name;
                    spec.name = spec.name.replace(/^sequential(\/|_\d+\/)/, '');
                    if (oldName !== spec.name) {
                        console.log(`✅ 權重視射修補: ${oldName} -> ${spec.name}`);
                    }
                });
            }

            return artifacts;
        };

        model = await tf.loadLayersModel(handler);
        confDetails.innerText = "🚀 銀河核心同步成功，模型已就緒";
        
        // 預熱張量運算
        tf.tidy(() => model.predict(tf.zeros([1, 28, 28, 1])));
    } catch (err) {
        console.error("載入失敗詳情:", err);
        confDetails.innerHTML = `<span style="color:#ff4757">❌ 載入失敗: ${err.message}</span>`;
    }
}

// ==========================================
// 3. 底層影像邏輯 (完全移植 p.py 的 OpenCV 演算法)
// ==========================================

/**
 * 手寫實作 cv2.dilate (膨脹)
 * 解決手寫線條太細在縮放後失真的問題
 */
function manualDilate(pixelData, width, height) {
    const output = new Uint8ClampedArray(pixelData.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let max = 0;
            // 3x3 核心
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

/**
 * 手寫實作 cv2.moments (質心偏移)
 * 這是 p.py 能精確辨識邊角數字的核心
 */
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

/**
 * 處理單一數字 ROI (對標 p.py 的 resize 與 centering)
 */
async function processDigitROI(roiCanvas) {
    const tempCtx = roiCanvas.getContext('2d');
    const raw = tempCtx.getImageData(0, 0, roiCanvas.width, roiCanvas.height);
    
    // 1. 轉灰階並應用二值化 (Threshold)
    let gray = new Uint8ClampedArray(raw.width * raw.height);
    for (let i = 0; i < raw.data.length; i += 4) {
        gray[i / 4] = raw.data[i] > 120 ? 255 : 0;
    }

    // 2. 膨脹處理 (Dilate)
    gray = manualDilate(gray, raw.width, raw.height);

    // 3. 計算質心位移
    const shift = getShiftVector(gray, raw.width, raw.height);

    // 4. 建立 28x28 畫布並進行對齊 (如同 p.py 的中心校正)
    const final = document.createElement('canvas');
    final.width = 28; final.height = 28;
    const fCtx = final.getContext('2d');
    fCtx.fillStyle = "black";
    fCtx.fillRect(0, 0, 28, 28);

    // 套用 p.py 的 45% Padding 邏輯進行縮放繪製
    const side = Math.max(roiCanvas.width, roiCanvas.height);
    const scale = (28 * (1 - MNIST_PAD)) / side;
    
    fCtx.save();
    fCtx.translate(14 + shift.dx * scale, 14 + shift.dy * scale);
    fCtx.scale(scale, scale);
    fCtx.drawImage(roiCanvas, -roiCanvas.width / 2, -roiCanvas.height / 2);
    fCtx.restore();

    // 5. 轉為張量預測
    const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(final, 1).toFloat().div(255.0).expandDims(0);
    });

    const pred = model.predict(tensor);
    const scores = await pred.data();
    const result = {
        digit: pred.argMax(-1).dataSync()[0],
        conf: Math.max(...scores)
    };

    tf.dispose([tensor, pred]);
    return result;
}

// ==========================================
// 4. 區域偵測與多位數掃描 (CCA 演算法)
// ==========================================

function findDigitRegions(ctx, isRealtime) {
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
                let minX = x, maxX = x, minY = y, maxY = y, pixels = 0;

                while (stack.length > 0) {
                    const [cx, cy] = stack.pop();
                    pixels++;
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
                // p.py 過濾雜訊與比例
                if (pixels * (step**2) < 200) continue;
                if (w / h > 2.5 || h / w > 3.2) continue;
                regions.push({ x: minX, y: minY, w, h });
            }
        }
    }
    return regions.sort((a, b) => a.x - b.x);
}

// ==========================================
// 5. 辨識執行與 UI 控制
// ==========================================

async function runRecognition(isRealtime = false) {
    if (!model) return;

    const snap = document.createElement('canvas');
    snap.width = canvas.width; snap.height = canvas.height;
    const sCtx = snap.getContext('2d');
    if (cameraStream) sCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    sCtx.drawImage(canvas, 0, 0);

    const regions = findDigitRegions(sCtx, isRealtime);
    let finalStr = "";
    let logHtml = "";

    if (isRealtime) ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < regions.length; i++) {
        const r = regions[i];
        const roi = document.createElement('canvas');
        roi.width = r.w; roi.height = r.h;
        roi.getContext('2d').putImageData(sCtx.getImageData(r.x, r.y, r.w, r.h), 0, 0);

        // 連體字切割 (p.py: width > height * 1.3)
        if (r.w > r.h * 1.35) {
            const mid = r.w / 2;
            const subs = [{ x: 0, w: mid }, { x: mid, w: r.w - mid }];
            for (const sub of subs) {
                const subC = document.createElement('canvas');
                subC.width = sub.w; subC.height = r.h;
                subC.getContext('2d').drawImage(roi, sub.x, 0, sub.w, r.h, 0, 0, sub.w, r.h);
                const res = await processDigitROI(subC);
                if (res.conf > 0.8) {
                    finalStr += res.digit;
                    logHtml += `區域 ${i}S: <span class="highlight">${res.digit}</span> (${(res.conf*100).toFixed(1)}%)<br>`;
                }
            }
        } else {
            const res = await processDigitROI(roi);
            if (res.conf >= (isRealtime ? 0.9 : 0.7)) {
                finalStr += res.digit;
                logHtml += `區域 ${i+1}: <span class="highlight">${res.digit}</span> (${(res.conf*100).toFixed(1)}%)<br>`;
                if (isRealtime) drawFocusBox(r, res.digit);
            }
        }
    }

    digitDisplay.innerText = finalStr || "---";
    confDetails.innerHTML = logHtml;
    if (isRealtime) updatePen();
}

function drawFocusBox(r, digit) {
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 3;
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.fillStyle = "#00FF00";
    ctx.font = "bold 20px Orbitron";
    ctx.fillText(digit, r.x, r.y - 8);
}

// ==========================================
// 6. 銀河效果與交互系統 (對標你原本的 JS)
// ==========================================

function spawnGalaxyEffect(x, y) {
    const star = document.createElement('div');
    star.className = "star-particle";
    const color = GALAXY_COLORS[Math.floor(Math.random() * GALAXY_COLORS.length)];
    star.style.cssText = `
        position: absolute; left: ${x}px; top: ${y}px;
        width: 5px; height: 5px; background: ${color};
        box-shadow: 0 0 12px ${color}; border-radius: 50%;
        pointer-events: none; animation: star-fade 0.8s forwards;
    `;
    document.body.appendChild(star);
    setTimeout(() => star.remove(), 800);
}

function clearUniverse() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    digitDisplay.innerText = "---";
    confDetails.innerText = "星域已回歸虛無";
    addNebula(20);
}

function addNebula(n) {
    for (let i = 0; i < n; i++) {
        ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.2})`;
        ctx.beginPath();
        ctx.arc(Math.random()*canvas.width, Math.random()*canvas.height, Math.random()*2, 0, Math.PI*2);
        ctx.fill();
    }
}

async function toggleCam() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        clearInterval(realtimeInterval);
        video.style.display = "none";
        mainBox.classList.remove('cam-active');
        camToggleBtn.innerHTML = "📷 開啟鏡頭";
        clearUniverse();
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = cameraStream;
            video.style.display = "block";
            mainBox.classList.add('cam-active');
            camToggleBtn.innerHTML = "📷 關閉鏡頭";
            realtimeInterval = setInterval(() => runRecognition(true), 500);
        } catch (e) { alert("鏡頭初始化失敗"); }
    }
}

// [基礎繪圖邏輯]
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

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true; ctx.beginPath();
    const p = getCoord(e); ctx.moveTo(p.x, p.y);
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const p = getCoord(e);
    ctx.lineTo(p.x, p.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.x, p.y);
    if (!isEraser) spawnGalaxyEffect(p.x + window.scrollX, p.y + window.scrollY);
});

const endDraw = () => { if (isDrawing) { isDrawing = false; if (!cameraStream) runRecognition(false); } };
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseleave', endDraw);
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); isDrawing = true; ctx.beginPath(); const p = getCoord(e); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if(isDrawing) { const p = getCoord(e); ctx.lineTo(p.x, p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x, p.y); } });
canvas.addEventListener('touchend', endDraw);

function toggleEraser() {
    isEraser = !isEraser;
    eraserBtn.innerText = isEraser ? "畫筆模式" : "橡皮擦模式";
    updatePen();
}

function handleUpload(e) {
    const reader = new FileReader();
    reader.onload = (ev) => {
        const img = new Image();
        img.onload = () => {
            clearUniverse();
            const s = Math.min(canvas.width/img.width, canvas.height/img.height) * 0.8;
            ctx.drawImage(img, (canvas.width-img.width*s)/2, (canvas.height-img.height*s)/2, img.width*s, img.height*s);
            runRecognition(false);
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
}

// 啟動系統
loadModelAndFix();
clearUniverse();
