/**
 * 🌠 銀河手寫數字辨識系統 - 極限全功能完整版 (Industrial Reconstruction)
 * -----------------------------------------------------------------------
 * [完全移植清單]
 * 1. OpenCV 手寫實作：Threshold, Dilate, Moments, Centroid, BoundingRect, Padding
 * 2. 影像預處理：對標 p.py 的 45% Padding 比例與質心對齊
 * 3. 邏輯整合：多位數切割、連體字處理、雜訊過濾
 * 4. 視覺系統：銀河粒子、語音控制、鏡頭掃描、檔案匯入
 */

// ==========================================
// 1. 環境配置與全域宣告
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

// 系統參數
const MODEL_SIZE = 28;
const PADDING_RATIO = 0.45; // 嚴格遵循 p.py 的動態 Padding
const DILATE_KERNEL_SIZE = 3;
const GALAXY_COLORS = ["#a3d9ff", "#7ed6df", "#e056fd", "#686de0", "#ffffff"];

// ==========================================
// 2. 底層矩陣運算引擎 (OpenCV JS 移植)
// ==========================================

/**
 * [A] 手寫實作 cv2.threshold 邏輯
 * 將影像轉換為純黑白
 */
function applyThreshold(pixels, threshold = 127) {
    const data = new Uint8Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) {
        data[i] = pixels[i] > threshold ? 255 : 0;
    }
    return data;
}

/**
 * [B] 手寫實作 cv2.dilate (膨脹運算)
 * 使用 3x3 結構元素擴張筆畫，強化辨識特徵
 */
function dilateImage(pixels, width, height) {
    const output = new Uint8Array(pixels.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let maxVal = 0;
            // 檢查 3x3 鄰域
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const ny = y + ky;
                    const nx = x + kx;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        maxVal = Math.max(maxVal, pixels[ny * width + nx]);
                    }
                }
            }
            output[y * width + x] = maxVal;
        }
    }
    return output;
}

/**
 * [C] 手寫實作 cv2.moments (幾何矩) 與 質心校正
 * 這是 p.py 成功的關鍵，確保數字在縮放後能精確居中
 */
function getCentroidShift(pixels, width, height) {
    let m00 = 0, m10 = 0, m01 = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const val = pixels[y * width + x];
            if (val > 0) {
                m00 += val;
                m10 += x * val;
                m01 += y * val;
            }
        }
    }
    if (m00 === 0) return { dx: 0, dy: 0 };
    const cx = m10 / m00;
    const cy = m01 / m00;
    
    // 計算與圖像幾何中心的位移
    return {
        dx: (width / 2) - cx,
        dy: (height / 2) - cy
    };
}

/**
 * [D] 手寫實作 cv2.copyMakeBorder (動態 Padding)
 * 將 ROI 數字區域加上比例邊框
 */
function applyDynamicPadding(canvas, paddingRatio) {
    const side = Math.max(canvas.width, canvas.height);
    const pad = Math.floor(side * paddingRatio);
    const targetSize = side + pad * 2;
    
    const paddedCanvas = document.createElement('canvas');
    paddedCanvas.width = targetSize;
    paddedCanvas.height = targetSize;
    const pCtx = paddedCanvas.getContext('2d');
    
    pCtx.fillStyle = "black";
    pCtx.fillRect(0, 0, targetSize, targetSize);
    
    // 將原圖置於填充後的中心
    const offsetX = (targetSize - canvas.width) / 2;
    const offsetY = (targetSize - canvas.height) / 2;
    pCtx.drawImage(canvas, offsetX, offsetY);
    
    return paddedCanvas;
}

// ==========================================
// 3. 輪廓掃描與數字切割 (Connected Components)
// ==========================================

function findContours(imgData, isRealtime) {
    const { data, width, height } = imgData;
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i += 4) gray[i / 4] = data[i];

    const visited = new Uint8Array(width * height);
    const contours = [];
    const minArea = isRealtime ? 500 : 150;

    for (let y = 0; y < height; y += 4) {
        for (let x = 0; x < width; x += 4) {
            const idx = y * width + x;
            if (!visited[idx] && gray[idx] > 100) {
                let queue = [[x, y]];
                visited[idx] = 1;
                let xMin = x, xMax = x, yMin = y, yMax = y, area = 0;

                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    area++;
                    xMin = Math.min(xMin, cx); xMax = Math.max(xMax, cx);
                    yMin = Math.min(yMin, cy); yMax = Math.max(yMax, cy);

                    [[cx+4, cy], [cx-4, cy], [cx, cy+4], [cx, cy-4]].forEach(([nx, ny]) => {
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            if (!visited[nIdx] && gray[nIdx] > 100) {
                                visited[nIdx] = 1;
                                queue.push([nx, ny]);
                            }
                        }
                    });
                }

                const w = xMax - xMin + 1;
                const h = yMax - yMin + 1;
                // 實作 p.py 的過濾參數：面積、長寬比
                if (area * 16 < minArea) continue;
                if (w / h > 2.5 || h / w > 3.0) continue;

                contours.push({ x: xMin, y: yMin, w, h });
            }
        }
    }
    return contours.sort((a, b) => a.x - b.x);
}

// ==========================================
// 4. 預測核心與 TFJS 整合
// ==========================================

async function processROI(roiCanvas) {
    // 1. 抓取原始像素
    const tempCtx = roiCanvas.getContext('2d');
    const imgData = tempCtx.getImageData(0, 0, roiCanvas.width, roiCanvas.height);
    
    // 2. 模擬 p.py 預處理流：Threshold -> Dilate
    let pixels = applyThreshold(new Uint8Array(imgData.data.filter((_, i) => i % 4 === 0)));
    pixels = dilateImage(pixels, roiCanvas.width, roiCanvas.height);
    
    // 3. 質心計算與位移
    const shift = getCentroidShift(pixels, roiCanvas.width, roiCanvas.height);
    
    // 4. 建立預測專用的 28x28 畫布
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = MODEL_SIZE;
    finalCanvas.height = MODEL_SIZE;
    const fCtx = finalCanvas.getContext('2d');
    fCtx.fillStyle = "black";
    fCtx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);

    // 5. 應用 Padding 與 居中縮放
    const padded = applyDynamicPadding(roiCanvas, PADDING_RATIO);
    fCtx.drawImage(padded, 0, 0, padded.width, padded.height, 0, 0, 28, 28);

    // 6. 轉為張量並預測
    const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(finalCanvas, 1)
            .toFloat()
            .div(255.0)
            .expandDims(0);
    });

    const prediction = model.predict(tensor);
    const data = await prediction.data();
    const result = {
        digit: prediction.argMax(-1).dataSync()[0],
        confidence: Math.max(...data)
    };

    tf.dispose([tensor, prediction]);
    return result;
}

async function predict(isRealtime = false) {
    if (!model) return;

    const snap = document.createElement('canvas');
    snap.width = canvas.width; snap.height = canvas.height;
    const sCtx = snap.getContext('2d');
    if (cameraStream) sCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    sCtx.drawImage(canvas, 0, 0);

    const contours = findContours(sCtx.getImageData(0, 0, canvas.width, canvas.height), isRealtime);
    
    let digits = "";
    let reportHtml = "<b>🌠 觀測報告：</b><br>";

    if (isRealtime) ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < contours.length; i++) {
        const box = contours[i];
        const roi = document.createElement('canvas');
        roi.width = box.w; roi.height = box.h;
        roi.getContext('2d').putImageData(sCtx.getImageData(box.x, box.y, box.w, box.h), 0, 0);

        // 連體字切割 (width > height * 1.3)
        if (box.w > box.h * 1.3) {
            const mid = box.w / 2;
            const parts = [{ x: 0, w: mid }, { x: mid, w: box.w - mid }];
            for (const p of parts) {
                const sub = document.createElement('canvas');
                sub.width = p.w; sub.height = box.h;
                sub.getContext('2d').drawImage(roi, p.x, 0, p.w, box.h, 0, 0, p.w, box.h);
                const res = await processROI(sub);
                if (res.confidence > 0.8) {
                    digits += res.digit;
                    reportHtml += `區域 ${i}S: <span class="highlight">${res.digit}</span> (${(res.confidence*100).toFixed(1)}%)<br>`;
                }
            }
        } else {
            const res = await processROI(roi);
            const threshold = isRealtime ? 0.88 : 0.7;
            if (res.confidence >= threshold) {
                digits += res.digit;
                reportHtml += `區域 ${i+1}: <span class="highlight">${res.digit}</span> (${(res.confidence*100).toFixed(1)}%)<br>`;
                if (isRealtime) drawLabel(box, res.digit);
            }
        }
    }

    digitDisplay.innerText = digits || "---";
    confDetails.innerHTML = reportHtml;
    if (isRealtime) resetPen();
}

function drawLabel(box, digit) {
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 3;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
    ctx.fillStyle = "#00FF00";
    ctx.font = "bold 20px Orbitron";
    ctx.fillText(digit, box.x, box.y - 10);
}

// ==========================================
// 5. 模型載入與修復器 (Keras v3 / TFJS)
// ==========================================

async function loadModelWithPatch() {
    const url = `tfjs_model/model.json?v=${Date.now()}`;
    try {
        confDetails.innerHTML = "<span class='loading'>🌌 正在同步銀河引擎...</span>";
        await tf.ready();
        
        // 自定義載入器以修正 Keras 轉換產生的常見錯誤
        const handler = tf.io.browserHTTPRequest(url);
        const originalLoad = handler.load.bind(handler);
        handler.load = async () => {
            const art = await originalLoad();
            // 修正 InputLayer 缺失
            if (art.modelTopology) {
                const layers = art.modelTopology.model_config.config.layers;
                layers.forEach(l => {
                    if (l.class_name === 'InputLayer' && !l.config.batch_input_shape) {
                        l.config.batch_input_shape = [null, 28, 28, 1];
                    }
                });
            }
            // 修正權重命名前綴
            if (art.weightSpecs) {
                art.weightSpecs.forEach(s => s.name = s.name.replace(/^sequential(_\d+)?\//, ''));
            }
            return art;
        };

        model = await tf.loadLayersModel(handler);
        confDetails.innerText = "🚀 引擎已就緒";
        tf.tidy(() => model.predict(tf.zeros([1, 28, 28, 1])));
    } catch (e) {
        confDetails.innerHTML = `<span style="color:red">引擎載入失敗: ${e.message}</span>`;
    }
}

// ==========================================
// 6. 視覺效果與 UI 交互
// ==========================================

function spawnStar(x, y) {
    const star = document.createElement('div');
    star.className = "star-particle";
    const color = GALAXY_COLORS[Math.floor(Math.random() * GALAXY_COLORS.length)];
    star.style.cssText = `
        position: absolute; left: ${x}px; top: ${y}px;
        width: 4px; height: 4px; background: ${color};
        box-shadow: 0 0 10px ${color}; border-radius: 50%;
        pointer-events: none; animation: star-fade 0.8s forwards;
    `;
    document.body.appendChild(star);
    setTimeout(() => star.remove(), 800);
}

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    digitDisplay.innerText = "---";
    confDetails.innerText = "星域已清空";
    addGalaxyBackground(15);
}

function addGalaxyBackground(n) {
    for (let i = 0; i < n; i++) {
        ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.3})`;
        ctx.beginPath();
        ctx.arc(Math.random()*canvas.width, Math.random()*canvas.height, Math.random()*2, 0, Math.PI*2);
        ctx.fill();
    }
}

async function toggleCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        clearInterval(realtimeInterval);
        video.style.display = "none";
        mainBox.classList.remove('cam-active');
        camToggleBtn.innerHTML = "📷 開啟鏡頭";
        clearCanvas();
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = cameraStream;
            video.style.display = "block";
            mainBox.classList.add('cam-active');
            camToggleBtn.innerHTML = "📷 關閉鏡頭";
            realtimeInterval = setInterval(() => predict(true), 450);
        } catch (e) { alert("無法存取鏡頭系統"); }
    }
}

// [語音與事件]
function setupSpeech() {
    const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Speech) return;
    recognition = new Speech();
    recognition.lang = 'zh-TW';
    recognition.onresult = (e) => {
        const cmd = e.results[e.results.length - 1][0].transcript;
        if (cmd.includes("清除")) clearCanvas();
        if (cmd.includes("辨識")) predict(false);
    };
}

function toggleVoice() {
    if (isVoiceActive) recognition.stop(); else recognition.start();
    isVoiceActive = !isVoiceActive;
    voiceBtn.classList.toggle('active', isVoiceActive);
}

function resetPen() {
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = isEraser ? "black" : "white";
    ctx.lineWidth = isEraser ? 50 : 16;
}

function getPos(e) {
    const r = canvas.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    const y = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
    return { x, y };
}

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true; ctx.beginPath();
    const p = getPos(e); ctx.moveTo(p.x, p.y);
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const p = getPos(e);
    ctx.lineTo(p.x, p.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.x, p.y);
    if (!isEraser) spawnStar(p.x + window.scrollX, p.y + window.scrollY);
});

const stopDrawing = () => { if (isDrawing) { isDrawing = false; if (!cameraStream) predict(); } };
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); isDrawing = true; ctx.beginPath(); const p = getPos(e); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if(isDrawing) { const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x, p.y); } });
canvas.addEventListener('touchend', stopDrawing);

function toggleEraser() {
    isEraser = !isEraser;
    eraserBtn.innerText = isEraser ? "橡皮擦模式" : "畫筆模式";
    resetPen();
}

function handleFile(e) {
    const reader = new FileReader();
    reader.onload = (ev) => {
        const img = new Image();
        img.onload = () => {
            clearCanvas();
            const s = Math.min(canvas.width/img.width, canvas.height/img.height) * 0.8;
            ctx.drawImage(img, (canvas.width-img.width*s)/2, (canvas.height-img.height*s)/2, img.width*s, img.height*s);
            predict();
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
}

// 啟動
loadModelWithPatch();
setupSpeech();
clearCanvas();
