/**
 * 銀河主題手寫數字辨識系統 - 完全移植版
 * 整合了 p.py 的影像處理邏輯與 script.js 的視覺特效
 */

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

// 全域變數
let model = null;
let isDrawing = false;
let isEraser = false;
let cameraStream = null;
let realtimeInterval = null;
let recognition = null;
let isVoiceActive = false;
let lastX = 0;
let lastY = 0;

// --- 1. 初始化與模型載入 ---

async function init() {
    try {
        // 更新顯示狀態
        digitDisplay.innerHTML = '<span class="pulse-icon">🌠</span>';
        confDetails.innerText = "正在連接銀河運算核心 (TF.js)...";
        
        // 載入 TensorFlow.js 模型
        // 請確保你的 tfjs_model 資料夾與 index.html 在同目錄
        model = await tf.loadLayersModel('tfjs_model/model.json');
        
        digitDisplay.innerText = "---";
        confDetails.innerText = "系統已就緒，請在畫布書寫";
    } catch (e) {
        console.error(e);
        digitDisplay.innerText = "❌";
        confDetails.innerText = "模型載入失敗，請確認 tfjs_model 資料夾路徑";
    }

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();
    addGalaxyEffects();
}

// --- 2. 影像處理核心 (移植自 p.py 的進階邏輯) ---

/**
 * 質心校正：對應 p.py 的 cv2.moments
 */
function getCentroid(data, width, height) {
    let m00 = 0, m10 = 0, m01 = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let val = data[y * width + x]; // 假設已正規化 0-1
            if (val > 0.1) {
                m00 += val;
                m10 += x * val;
                m01 += y * val;
            }
        }
    }
    if (m00 === 0) return { cx: 14, cy: 14 };
    return { cx: m10 / m00, cy: m01 / m00 };
}

/**
 * 進階預處理：對應 p.py 的 advanced_preprocess
 */
function advancedPreprocess(roiCanvas) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(roiCanvas, 1).toFloat();

        // A. 強化筆畫 (對應 p.py 的 cv2.dilate)
        // 使用 2x2 MaxPool 來模擬膨脹效果
        tensor = tf.dilation2d(tensor.expandDims(0), tf.ones([2, 2, 1]), [1, 1, 1, 1], 'same').squeeze(0);

        // B. 動態 Padding (對應 p.py 的 copyMakeBorder)
        const h = tensor.shape[0];
        const w = tensor.shape[1];
        const pad = Math.floor(Math.max(h, w) * 0.45);
        tensor = tensor.pad([[pad, pad], [pad, pad], [0, 0]]);

        // C. 縮放至 28x28 (對應 p.py 的 cv2.resize)
        tensor = tf.image.resizeBilinear(tensor, [28, 28]);

        // D. 質心校正 (對應 p.py 的 warpAffine)
        const dataSync = tensor.dataSync();
        const { cx, cy } = getCentroid(dataSync, 28, 28);
        const tx = 14 - cx;
        const ty = 14 - cy;
        // 平移矩陣轉換
        tensor = tf.image.transform(tensor.expandDims(0), [1, 0, -tx, 0, 1, -ty, 0, 0], 'bilinear').squeeze(0);

        // E. 正規化 (對應 p.py 的 / 255.0)
        return tensor.div(255.0).expandDims(0);
    });
}

/**
 * 影像清洗與區域偵測：對應 p.py 的 connectedComponentsWithStats
 */
function findComponents(imageData, isRealtime) {
    const { width, height, data } = imageData;
    const minArea = isRealtime ? 500 : 150;
    const visited = new Uint8Array(width * height);
    const comps = [];

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            // 只看亮色像素 (二值化)
            if (data[idx * 4] > 100 && !visited[idx]) {
                // BFS 找連通區域
                const q = [[x, y]];
                visited[idx] = 1;
                let minX = x, maxX = x, minY = y, maxY = y;
                let area = 0;
                const pixels = [];

                while (q.length > 0) {
                    const [cx, cy] = q.shift();
                    area++;
                    pixels.push([cx, cy]);
                    if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
                    if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;

                    [[0,1],[0,-1],[1,0],[-1,0]].forEach(([dx, dy]) => {
                        const nx = cx + dx, ny = cy + dy;
                        const nidx = ny * width + nx;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height && 
                            data[nidx * 4] > 100 && !visited[nidx]) {
                            visited[nidx] = 1;
                            q.push([nx, ny]);
                        }
                    });
                }

                const w = maxX - minX + 1;
                const h = maxY - minY + 1;
                const aspectRatio = w / h;
                const solidity = area / (w * h);

                // --- 移植 p.py 的過濾邏輯 ---
                if (area < minArea) continue;
                if (aspectRatio > 2.5 || aspectRatio < 0.15) continue;
                if (solidity < 0.15) continue;
                
                // 邊緣過濾
                const border = 8;
                if (minX < border || minY < border || maxX > (width - border) || maxY > (height - border)) {
                    if (area < 1000) continue;
                }

                comps.push({ x: minX, y: minY, w, h, area, pixels });
            }
        }
    }
    return comps.sort((a, b) => a.x - b.x); // 從左到右排序
}

// --- 3. 預測核心功能 ---

async function predict(isRealtime = false) {
    if (!model) return;

    // 建立快照畫布
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext('2d');

    // 如果相機開著，抓取相機畫面；否則只抓畫布
    if (cameraStream) {
        tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }
    tCtx.drawImage(canvas, 0, 0);

    const imageData = tCtx.getImageData(0, 0, canvas.width, canvas.height);
    
    // 1. 背景反轉檢測 (對應 p.py 的 255 - gray)
    let avgBrightness = 0;
    for (let i = 0; i < imageData.data.length; i += 4) {
        avgBrightness += imageData.data[i];
    }
    avgBrightness /= (imageData.width * imageData.height);
    
    // 如果背景太亮，反轉它以便辨識
    if (avgBrightness > 120) {
        for (let i = 0; i < imageData.data.length; i += 4) {
            imageData.data[i] = 255 - imageData.data[i];
            imageData.data[i+1] = 255 - imageData.data[i+1];
            imageData.data[i+2] = 255 - imageData.data[i+2];
        }
    }

    // 2. 影像清洗
    const comps = findComponents(imageData, isRealtime);
    
    let finalRes = "";
    let details = [];
    let validBoxes = [];

    for (let comp of comps) {
        // 建立 ROI 畫布
        const roiCanvas = document.createElement('canvas');
        roiCanvas.width = comp.w;
        roiCanvas.height = comp.h;
        const rCtx = roiCanvas.getContext('2d');
        rCtx.fillStyle = "black";
        rCtx.fillRect(0, 0, comp.w, comp.h);
        rCtx.fillStyle = "white";
        // 只畫出該連通區域的像素 (清洗雜訊)
        comp.pixels.forEach(([px, py]) => {
            rCtx.fillRect(px - comp.x, py - comp.y, 1, 1);
        });

        // 3. 連體字切割 (對應 p.py 的 w > h * 1.3)
        if (comp.w > comp.h * 1.3) {
            const splitX = Math.floor(comp.w / 2); // 簡化版切割
            const rois = [
                {x: 0, w: splitX},
                {x: splitX, w: comp.w - splitX}
            ];
            for(let r of rois) {
                const subCanvas = document.createElement('canvas');
                subCanvas.width = r.w; subCanvas.height = comp.h;
                subCanvas.getContext('2d').drawImage(roiCanvas, r.x, 0, r.w, comp.h, 0, 0, r.w, comp.h);
                
                const tensor = advancedPreprocess(subCanvas);
                const pred = model.predict(tensor);
                const score = pred.dataSync();
                const digit = pred.argMax(1).dataSync()[0];
                const conf = Math.max(...score);
                if (conf > 0.8) {
                    finalRes += digit;
                    details.push({ digit, conf: (conf * 100).toFixed(1) + "%" });
                }
            }
        } else {
            // 4. 一般數字預測
            const tensor = advancedPreprocess(roiCanvas);
            const pred = model.predict(tensor);
            const score = pred.dataSync();
            const digit = pred.argMax(1).dataSync()[0];
            const conf = Math.max(...score);

            if (isRealtime && conf < 0.85) continue;

            finalRes += digit;
            details.push({ digit, conf: (conf * 100).toFixed(1) + "%" });
            validBoxes.push(comp);
        }
    }

    // 更新介面
    if (!isRealtime || finalRes !== "") {
        digitDisplay.innerText = finalRes || "---";
        updateDetails({ details });
        if (finalRes !== "") addVisualFeedback("#2ecc71");
    }

    // 如果是即時模式，在畫面上畫框
    if (cameraStream) {
        drawRealtimeBoxes(validBoxes, details);
    }
}

function drawRealtimeBoxes(boxes, details) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    boxes.forEach((box, i) => {
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.fillStyle = "#00FF00";
        ctx.font = "bold 24px Orbitron";
        ctx.fillText(details[i] ? details[i].digit : "", box.x, box.y - 5);
    });
    updatePen();
}

// --- 4. 原始 script.js 的所有介面與特效邏輯 (完整保留) ---

function updatePen() {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (isEraser) {
        ctx.strokeStyle = "black";
        ctx.lineWidth = 40;
    } else {
        ctx.strokeStyle = "white";
        ctx.lineWidth = 15;
    }
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
    if (!cameraStream) {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    digitDisplay.innerText = "---";
    confDetails.innerText = "畫布已清空，銀河已淨空";
    addVisualFeedback("#2ecc71");
    addGalaxyEffects();
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
            realtimeInterval = setInterval(() => predict(true), 400);
            clearCanvas();
            addVisualFeedback("#9b59b6");
        } catch (err) {
            alert("鏡頭啟動失敗: " + err);
        }
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    if (realtimeInterval) clearInterval(realtimeInterval);
    video.style.display = "none";
    mainBox.classList.remove('cam-active');
    camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 開啟鏡頭';
    init();
}

// 繪圖事件 (滑鼠)
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// 觸控支援
canvas.addEventListener('touchstart', (e) => { if(e.touches.length === 1) startDrawing(e); });
canvas.addEventListener('touchmove', (e) => { if(e.touches.length === 1) draw(e); });
canvas.addEventListener('touchend', stopDrawing);

function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
}

function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getCanvasCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    lastX = x; lastY = y;
    if (!isEraser) addDrawingEffect(x, y);
}

function draw(e) {
    e.preventDefault();
    if (!isDrawing) return;
    const { x, y } = getCanvasCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    if (!isEraser) addDrawingEffect(x, y);
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
        if (!cameraStream) setTimeout(() => predict(), 100);
    }
}

function addDrawingEffect(x, y) {
    const effect = document.createElement('div');
    effect.className = 'drawing-effect';
    effect.style.left = x + 'px';
    effect.style.top = y + 'px';
    // 這裡我們直接寫 style 確保不依賴外部 CSS 的效果
    Object.assign(effect.style, {
        position: 'absolute', width: '8px', height: '8px',
        borderRadius: '50%', background: '#a3d9ff',
        pointerEvents: 'none', zIndex: '1000', opacity: '0.8'
    });
    mainBox.appendChild(effect);
    setTimeout(() => effect.remove(), 500);
}

function addVisualFeedback(color) {
    const box = document.querySelector('.canvas-box');
    box.style.boxShadow = `0 0 40px ${color}`;
    setTimeout(() => box.style.boxShadow = '', 400);
}

function updateDetails(data) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (!data.details || data.details.length === 0) {
        html += "等待有效數字入鏡...";
    } else {
        data.details.forEach((item, i) => {
            const color = i % 2 === 0 ? "#a3d9ff" : "#ff6b9d";
            html += `數字 ${i + 1}: <b style="color:${color}">${item.digit}</b> (信心度: ${item.conf})<br>`;
        });
    }
    confDetails.innerHTML = html;
}

// 語音辨識 (完整保留)
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-TW';
    recognition.continuous = true;
    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        if (transcript.includes('清除')) clearCanvas();
        else if (transcript.includes('辨識')) predict();
        else if (transcript.includes('鏡頭')) toggleCamera();
        else if (transcript.includes('橡皮擦')) toggleEraser();
    };
}

function toggleVoice() {
    if (!recognition) return alert("瀏覽器不支援語音");
    isVoiceActive = !isVoiceActive;
    if (isVoiceActive) {
        recognition.start();
        voiceBtn.innerHTML = '🌌 語音：開啟';
        voiceBtn.classList.add('voice-active');
        voiceStatus.style.display = 'block';
    } else {
        recognition.stop();
        voiceBtn.innerHTML = '🌌 語音：關閉';
        voiceBtn.classList.remove('voice-active');
        voiceStatus.style.display = 'none';
    }
}

function addGalaxyEffects() {
    // 畫布背景的小星星
    ctx.fillStyle = "rgba(163, 217, 255, 0.3)";
    ctx.beginPath(); ctx.arc(650, 20, 2, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(30, 300, 2, 0, Math.PI * 2); ctx.fill();
}

function triggerFile() { fileInput.click(); }

function handleFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            if (cameraStream) stopCamera();
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            const ratio = Math.min(canvas.width / img.width, canvas.height / img.height) * 0.8;
            const w = img.width * ratio, h = img.height * ratio;
            ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
            predict();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// 啟動系統
init();