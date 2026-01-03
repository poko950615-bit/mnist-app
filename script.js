/**
 * 🌌 銀河手寫數字辨識系統 - 完整移植版
 * 整合 p.py 的影像處理邏輯與完整的 UI 功能
 */

// --- 1. 元素選取與變數設定 ---
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

// TensorFlow.js 模型
let model = null;
let isDrawing = false;
let isEraser = false;
let cameraStream = null;
let realtimeInterval = null;
let lastX = 0;
let lastY = 0;
let recognition = null;
let isVoiceActive = false;

// --- 2. 模型修復載入器 (解決 Keras v3 相容性) ---
class PatchModelLoader {
    constructor(url) {
        this.url = url;
    }

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

// --- 3. 影像處理核心 (移植自 p.py) ---

// 高斯模糊函數
function gaussianBlur(imageData, radius = 2) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const result = new ImageData(width, height);
    
    const kernelSize = radius * 2 + 1;
    const kernel = [];
    const sigma = radius / 2;
    let sum = 0;
    
    // 創建高斯核
    for (let x = -radius; x <= radius; x++) {
        for (let y = -radius; y <= radius; y++) {
            const value = Math.exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * Math.PI * sigma * sigma);
            kernel.push(value);
            sum += value;
        }
    }
    
    // 正規化核
    kernel.forEach((val, idx) => kernel[idx] = val / sum);
    
    // 應用卷積
    for (let y = radius; y < height - radius; y++) {
        for (let x = radius; x < width - radius; x++) {
            let r = 0, g = 0, b = 0;
            let kIdx = 0;
            
            for (let ky = -radius; ky <= radius; ky++) {
                for (let kx = -radius; kx <= radius; kx++) {
                    const pixelIdx = ((y + ky) * width + (x + kx)) * 4;
                    const weight = kernel[kIdx++];
                    
                    r += data[pixelIdx] * weight;
                    g += data[pixelIdx + 1] * weight;
                    b += data[pixelIdx + 2] * weight;
                }
            }
            
            const resultIdx = (y * width + x) * 4;
            result.data[resultIdx] = r;
            result.data[resultIdx + 1] = g;
            result.data[resultIdx + 2] = b;
            result.data[resultIdx + 3] = data[resultIdx + 3];
        }
    }
    
    return result;
}

// Otsu 閾值二值化
function otsuThreshold(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    
    // 計算直方圖
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i += 4) {
        const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
        histogram[Math.floor(gray)]++;
    }
    
    // Otsu 算法
    let total = width * height;
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * histogram[i];
    
    let sumB = 0;
    let wB = 0;
    let wF = 0;
    let maxVariance = 0;
    let threshold = 0;
    
    for (let i = 0; i < 256; i++) {
        wB += histogram[i];
        if (wB === 0) continue;
        
        wF = total - wB;
        if (wF === 0) break;
        
        sumB += i * histogram[i];
        
        let mB = sumB / wB;
        let mF = (sum - sumB) / wF;
        
        let variance = wB * wF * (mB - mF) * (mB - mF);
        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = i;
        }
    }
    
    // 應用閾值
    const result = new ImageData(width, height);
    for (let i = 0; i < data.length; i += 4) {
        const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
        const binary = gray > threshold ? 255 : 0;
        
        result.data[i] = binary;
        result.data[i + 1] = binary;
        result.data[i + 2] = binary;
        result.data[i + 3] = 255;
    }
    
    return { threshold, result };
}

// 連通域分析
function connectedComponentsWithStats(binaryData) {
    const width = binaryData.width;
    const height = binaryData.height;
    const data = binaryData.data;
    
    const visited = new Array(width * height).fill(false);
    const labels = new Array(width * height).fill(-1);
    const stats = [];
    let currentLabel = 0;
    
    const dx = [-1, 0, 1, -1, 1, -1, 0, 1];
    const dy = [-1, -1, -1, 0, 0, 1, 1, 1];
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            
            if (!visited[idx] && data[idx * 4] === 255) {
                // BFS 搜尋連通域
                const queue = [[x, y]];
                visited[idx] = true;
                labels[idx] = currentLabel;
                
                let minX = x, maxX = x, minY = y, maxY = y;
                let area = 0;
                
                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    area++;
                    
                    minX = Math.min(minX, cx);
                    maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy);
                    maxY = Math.max(maxY, cy);
                    
                    // 檢查8鄰居
                    for (let d = 0; d < 8; d++) {
                        const nx = cx + dx[d];
                        const ny = cy + dy[d];
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            
                            if (!visited[nIdx] && data[nIdx * 4] === 255) {
                                visited[nIdx] = true;
                                labels[nIdx] = currentLabel;
                                queue.push([nx, ny]);
                            }
                        }
                    }
                }
                
                stats.push({
                    x: minX,
                    y: minY,
                    w: maxX - minX + 1,
                    h: maxY - minY + 1,
                    area: area
                });
                
                currentLabel++;
            }
        }
    }
    
    return {
        num: currentLabel,
        labels,
        stats,
        visited
    };
}

// 膨脹操作
function dilate(imageData, kernelSize = 2) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const result = new ImageData(width, height);
    
    const half = Math.floor(kernelSize / 2);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let maxVal = 0;
            
            for (let ky = -half; ky <= half; ky++) {
                for (let kx = -half; kx <= half; kx++) {
                    const nx = x + kx;
                    const ny = y + ky;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const idx = (ny * width + nx) * 4;
                        maxVal = Math.max(maxVal, data[idx]);
                    }
                }
            }
            
            const idx = (y * width + x) * 4;
            result.data[idx] = maxVal;
            result.data[idx + 1] = maxVal;
            result.data[idx + 2] = maxVal;
            result.data[idx + 3] = 255;
        }
    }
    
    return result;
}

// 進階預處理 (移植自 p.py 的 advanced_preprocess)
function advancedPreprocess(roiCanvas) {
    const roiCtx = roiCanvas.getContext('2d');
    let imageData = roiCtx.getImageData(0, 0, roiCanvas.width, roiCanvas.height);
    
    // 1. 膨脹
    const dilated = dilate(imageData, 2);
    
    // 2. 動態 Padding
    const h = roiCanvas.height;
    const w = roiCanvas.width;
    const pad = Math.floor(Math.max(h, w) * 0.45);
    
    const paddedCanvas = document.createElement('canvas');
    paddedCanvas.width = w + 2 * pad;
    paddedCanvas.height = h + 2 * pad;
    const paddedCtx = paddedCanvas.getContext('2d');
    
    // 填充黑色背景
    paddedCtx.fillStyle = 'black';
    paddedCtx.fillRect(0, 0, paddedCanvas.width, paddedCanvas.height);
    
    // 畫上原始影像
    paddedCtx.putImageData(dilated, pad, pad);
    
    // 3. 縮放至 28x28
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = 28;
    resizedCanvas.height = 28;
    const resizedCtx = resizedCanvas.getContext('2d');
    
    resizedCtx.drawImage(paddedCanvas, 0, 0, 28, 28);
    
    // 4. 質心校正
    const resizedData = resizedCtx.getImageData(0, 0, 28, 28);
    let sumX = 0, sumY = 0, total = 0;
    
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const idx = (y * 28 + x) * 4;
            const val = resizedData.data[idx] / 255;
            sumX += x * val;
            sumY += y * val;
            total += val;
        }
    }
    
    if (total > 0) {
        const cx = sumX / total;
        const cy = sumY / total;
        const dx = 14 - cx;
        const dy = 14 - cy;
        
        const correctedCanvas = document.createElement('canvas');
        correctedCanvas.width = 28;
        correctedCanvas.height = 28;
        const correctedCtx = correctedCanvas.getContext('2d');
        
        correctedCtx.translate(dx, dy);
        correctedCtx.drawImage(resizedCanvas, 0, 0);
        
        return correctedCanvas;
    }
    
    return resizedCanvas;
}

// --- 4. 主辨識函數 ---
async function predict(isRealtime = false) {
    if (!model) {
        digitDisplay.innerText = "❌";
        confDetails.innerHTML = "<b>錯誤：</b>模型尚未載入";
        return;
    }
    
    try {
        // 獲取畫布影像
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        
        if (cameraStream) {
            tempCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }
        tempCtx.drawImage(canvas, 0, 0);
        
        // 轉為灰階
        let imageData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
        const grayData = new ImageData(canvas.width, canvas.height);
        
        // 計算平均亮度
        let sum = 0;
        for (let i = 0; i < imageData.data.length; i += 4) {
            const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
            sum += gray;
        }
        const avgBrightness = sum / (imageData.data.length / 4);
        
        // 背景反轉檢測
        for (let i = 0; i < imageData.data.length; i += 4) {
            let gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
            
            if (avgBrightness > 120) {
                gray = 255 - gray;
            }
            
            grayData.data[i] = gray;
            grayData.data[i + 1] = gray;
            grayData.data[i + 2] = gray;
            grayData.data[i + 3] = 255;
        }
        
        // 去噪與二值化
        const blurred = gaussianBlur(grayData, 2);
        const { result: threshData } = otsuThreshold(blurred);
        
        // 影像清洗機制
        const { num, stats, visited } = connectedComponentsWithStats(threshData);
        
        // 建立乾淨的底圖
        const cleanedCanvas = document.createElement('canvas');
        cleanedCanvas.width = canvas.width;
        cleanedCanvas.height = canvas.height;
        const cleanedCtx = cleanedCanvas.getContext('2d');
        cleanedCtx.fillStyle = 'black';
        cleanedCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        const comps = [];
        const validBoxes = [];
        const MIN_AREA = isRealtime ? 500 : 150;
        
        // 過濾連通域
        for (let i = 0; i < num; i++) {
            const { x, y, w, h, area } = stats[i];
            
            // 1. 面積過小
            if (area < MIN_AREA) continue;
            
            // 2. 排除過於細長或寬大的線條
            const aspectRatio = w / h;
            if (aspectRatio > 2.5 || aspectRatio < 0.15) continue;
            
            // 3. Solidity (填滿率) 檢查
            const rectArea = w * h;
            if (area / rectArea < 0.15) continue;
            
            // 4. 邊緣無效區過濾
            const border = 8;
            if (x < border || y < border || 
                (x + w) > (canvas.width - border) || 
                (y + h) > (canvas.height - border)) {
                if (area < 1000) continue;
            }
            
            // 通過檢查，畫回清洗後的底圖
            const roiCtx = cleanedCanvas.getContext('2d');
            roiCtx.fillStyle = 'white';
            
            // 畫出這個連通域
            for (let py = y; py < y + h; py++) {
                for (let px = x; px < x + w; px++) {
                    const idx = py * canvas.width + px;
                    if (visited[idx] === i) {
                        roiCtx.fillRect(px, py, 1, 1);
                    }
                }
            }
            
            comps.push({ x, y, w, h });
        }
        
        // 排序 (由左至右)
        comps.sort((a, b) => a.x - b.x);
        
        let finalRes = "";
        const details = [];
        
        // 對每個區域進行辨識
        for (const { x, y, w, h } of comps) {
            // 提取 ROI
            const roiCanvas = document.createElement('canvas');
            roiCanvas.width = w;
            roiCanvas.height = h;
            const roiCtx = roiCanvas.getContext('2d');
            
            const roiImageData = cleanedCtx.getImageData(x, y, w, h);
            roiCtx.putImageData(roiImageData, 0, 0);
            
            // 連體字切割邏輯
            if (w > h * 1.3) {
                // 水平投影
                const proj = new Array(w).fill(0);
                const roiData = roiCtx.getImageData(0, 0, w, h);
                
                for (let px = 0; px < w; px++) {
                    for (let py = 0; py < h; py++) {
                        const idx = (py * w + px) * 4;
                        if (roiData.data[idx] > 128) {
                            proj[px]++;
                        }
                    }
                }
                
                // 找到分割點
                const start = Math.floor(w * 0.3);
                const end = Math.floor(w * 0.7);
                let minVal = h + 1;
                let splitX = start;
                
                for (let px = start; px < end; px++) {
                    if (proj[px] < minVal) {
                        minVal = proj[px];
                        splitX = px;
                    }
                }
                
                // 分割成兩個子區域
                const subRois = [
                    { x: 0, y: 0, w: splitX, h: h },
                    { x: splitX, y: 0, w: w - splitX, h: h }
                ];
                
                for (const subRoi of subRois) {
                    if (subRoi.w < 5) continue;
                    
                    const subCanvas = document.createElement('canvas');
                    subCanvas.width = subRoi.w;
                    subCanvas.height = subRoi.h;
                    const subCtx = subCanvas.getContext('2d');
                    
                    subCtx.drawImage(roiCanvas, subRoi.x, subRoi.y, subRoi.w, subRoi.h, 
                                    0, 0, subRoi.w, subRoi.h);
                    
                    // 進階預處理
                    const processedCanvas = advancedPreprocess(subCanvas);
                    
                    // 轉為 Tensor
                    const tensor = tf.browser.fromPixels(processedCanvas, 1)
                        .toFloat()
                        .div(tf.scalar(255))
                        .reshape([1, 28, 28, 1]);
                    
                    // 預測
                    const pred = model.predict(tensor);
                    const predData = await pred.data();
                    const digit = pred.argMax(-1).dataSync()[0];
                    const confidence = Math.max(...predData);
                    
                    if (confidence > 0.8) {
                        finalRes += digit.toString();
                        details.push({
                            digit: digit,
                            conf: `${(confidence * 100).toFixed(1)}%`
                        });
                    }
                    
                    tensor.dispose();
                    pred.dispose();
                }
                
                continue;
            }
            
            // 一般數字預測
            const processedCanvas = advancedPreprocess(roiCanvas);
            
            // 轉為 Tensor
            const tensor = tf.browser.fromPixels(processedCanvas, 1)
                .toFloat()
                .div(tf.scalar(255))
                .reshape([1, 28, 28, 1]);
            
            // 預測
            const pred = model.predict(tensor);
            const predData = await pred.data();
            const digit = pred.argMax(-1).dataSync()[0];
            const confidence = Math.max(...predData);
            
            // 信心度過濾
            if (isRealtime && confidence < 0.85) {
                tensor.dispose();
                pred.dispose();
                continue;
            }
            
            finalRes += digit.toString();
            details.push({
                digit: digit,
                conf: `${(confidence * 100).toFixed(1)}%`
            });
            
            validBoxes.push({
                x: x,
                y: y,
                w: w,
                h: h
            });
            
            tensor.dispose();
            pred.dispose();
        }
        
        // 更新顯示
        digitDisplay.innerText = finalRes || "---";
        updateDetails(details);
        
        // 如果是即時模式，畫出偵測框
        if (isRealtime && cameraStream && validBoxes.length > 0) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            validBoxes.forEach((box, index) => {
                // 畫綠色框框
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.strokeRect(box.x, box.y, box.w, box.h);
                
                // 畫辨識到的數字
                const detectedDigit = details[index] ? details[index].digit : "";
                ctx.fillStyle = "#00FF00";
                ctx.font = "bold 24px Arial";
                ctx.fillText(detectedDigit, box.x, box.y - 5);
            });
            
            updatePen();
        }
        
        return {
            full_digit: finalRes,
            details: details,
            boxes: validBoxes
        };
        
    } catch (err) {
        console.error("辨識錯誤:", err);
        digitDisplay.innerText = "❌";
        confDetails.innerHTML = `<b>錯誤：</b>${err.message}`;
        return { error: err.message };
    }
}

// --- 5. UI 互動功能 ---

// 初始化
async function init() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();
    
    // 載入 TensorFlow.js 模型
    try {
        confDetails.innerText = "🌌 正在載入銀河辨識引擎...";
        
        // 優先使用 CPU 確保穩定
        await tf.setBackend('cpu');
        await tf.ready();
        
        // 載入模型
        const modelUrl = 'tfjs_model/model.json';
        model = await tf.loadLayersModel(new PatchModelLoader(modelUrl));
        
        console.log("✅ 模型載入成功！");
        confDetails.innerText = "🚀 系統就緒，請開始在星域書寫";
        
        // 模型暖身
        tf.tidy(() => {
            model.predict(tf.zeros([1, 28, 28, 1]));
        });
        
    } catch (err) {
        console.error("模型載入失敗:", err);
        confDetails.innerHTML = `<span style="color: #ff4d4d">❌ 錯誤: ${err.message}</span>`;
    }
}

// 更新畫筆設定
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

// 切換橡皮擦
function toggleEraser() {
    isEraser = !isEraser;
    eraserBtn.innerText = isEraser ? "橡皮擦：開啟" : "橡皮擦：關閉";
    eraserBtn.classList.toggle('eraser-active', isEraser);
    updatePen();
    
    // 視覺回饋
    if (isEraser) {
        addVisualFeedback("#e74c3c");
    }
}

// 清除畫布
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!cameraStream) {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    digitDisplay.innerText = "---";
    confDetails.innerText = "畫布已清空，銀河已淨空";
    addVisualFeedback("#2ecc71");
}

// 視覺回饋效果
function addVisualFeedback(color) {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
        const originalBoxShadow = btn.style.boxShadow;
        btn.style.boxShadow = `0 0 20px ${color}`;
        
        setTimeout(() => {
            btn.style.boxShadow = originalBoxShadow;
        }, 300);
    });
}

// 切換相機
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
            
            realtimeInterval = setInterval(() => {
                predict(true);
            }, 400);
            
            clearCanvas();
            addVisualFeedback("#9b59b6");
        } catch (err) {
            alert("鏡頭啟動失敗: " + err);
        }
    }
}

// 停止相機
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
    addVisualFeedback("#34495e");
}

// 觸發檔案選擇
function triggerFile() {
    fileInput.click();
    addVisualFeedback("#3498db");
}

// 處理檔案上傳
function handleFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (cameraStream) stopCamera();
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            const ratio = Math.min(canvas.width / img.width, canvas.height / img.height) * 0.8;
            const w = img.width * ratio;
            const h = img.height * ratio;
            ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
            predict(false);
            addVisualFeedback("#3498db");
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// 更新詳細資訊
function updateDetails(data) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (!data || data.length === 0) {
        html += "等待有效數字入鏡...";
    } else {
        data.forEach((item, i) => {
            const color = i % 2 === 0 ? "#a3d9ff" : "#ff6b9d";
            html += `數字 ${i + 1}: <b style="color:${color}">${item.digit}</b> (信心度: ${item.conf})<br>`;
        });
    }
    confDetails.innerHTML = html;
}

// --- 6. 語音功能 ---

// 初始化語音辨識
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        voiceBtn.style.display = 'none';
        return;
    }
    
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-TW';
    recognition.continuous = true;
    recognition.interimResults = false;
    
    recognition.onstart = () => {
        isVoiceActive = true;
        updateVoiceButton();
        voiceStatus.style.display = 'block';
        addVisualFeedback("#ff6b9d");
    };
    
    recognition.onend = () => {
        if (isVoiceActive) {
            try {
                recognition.start();
            } catch (e) {
                console.log("語音識別重啟失敗:", e);
                isVoiceActive = false;
                updateVoiceButton();
                voiceStatus.style.display = 'none';
            }
        } else {
            updateVoiceButton();
            voiceStatus.style.display = 'none';
        }
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        
        if (transcript.includes('清除') || transcript.includes('清空')) {
            clearCanvas();
        } else if (transcript.includes('開始') || transcript.includes('辨識')) {
            predict(false);
        } else if (transcript.includes('鏡頭') || transcript.includes('相機')) {
            toggleCamera();
        } else if (transcript.includes('橡皮擦')) {
            toggleEraser();
        } else {
            digitDisplay.innerText = transcript;
            confDetails.innerHTML = `<b>語音來源：</b><span style="color:#ff6b9d">${transcript}</span>`;
            addVisualFeedback("#ff6b9d");
        }
    };
    
    recognition.onerror = (event) => {
        console.log("語音識別錯誤:", event.error);
        if (event.error === 'not-allowed' || event.error === 'audio-capture') {
            alert("請允許瀏覽器使用麥克風權限");
            isVoiceActive = false;
            updateVoiceButton();
            voiceStatus.style.display = 'none';
        }
    };
}

// 更新語音按鈕狀態
function updateVoiceButton() {
    if (isVoiceActive) {
        voiceBtn.innerHTML = '<span class="btn-icon">🌌</span> 語音輸入：開啟';
        voiceBtn.classList.add('voice-active');
    } else {
        voiceBtn.innerHTML = '<span class="btn-icon">🌌</span> 語音輸入：關閉';
        voiceBtn.classList.remove('voice-active');
    }
}

// 切換語音輸入
function toggleVoice() {
    if (!recognition) {
        alert("您的瀏覽器不支援語音識別功能");
        return;
    }
    
    if (isVoiceActive) {
        isVoiceActive = false;
        recognition.stop();
        updateVoiceButton();
        voiceStatus.style.display = 'none';
        addVisualFeedback("#34495e");
    } else {
        try {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    stream.getTracks().forEach(track => track.stop());
                    recognition.start();
                    updateVoiceButton();
                    addVisualFeedback("#ff6b9d");
                })
                .catch(err => {
                    console.log("麥克風權限錯誤:", err);
                    alert("請允許使用麥克風以啟用語音輸入功能");
                });
        } catch (e) {
            console.log("語音識別啟動錯誤:", e);
            alert("無法啟動語音識別功能");
        }
    }
}

// --- 7. 繪圖事件處理 ---

// 獲取畫布座標
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

// 開始繪圖
function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getCanvasCoordinates(e);
    
    ctx.beginPath();
    ctx.moveTo(x, y);
    
    lastX = x;
    lastY = y;
}

// 繪圖中
function draw(e) {
    e.preventDefault();
    
    if (!isDrawing) return;
    
    const { x, y } = getCanvasCoordinates(e);
    
    ctx.lineTo(x, y);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(x, y);
    
    lastX = x;
    lastY = y;
}

// 停止繪圖
function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
        if (!cameraStream) {
            setTimeout(() => predict(false), 100);
        }
    }
}

// 處理觸控開始
function handleTouchStart(e) {
    if (e.touches.length === 1) {
        startDrawing(e);
    }
}

// 處理觸控移動
function handleTouchMove(e) {
    if (e.touches.length === 1) {
        draw(e);
    }
}

// --- 8. 事件監聽器 ---

// 滑鼠事件
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// 觸控事件
canvas.addEventListener('touchstart', handleTouchStart);
canvas.addEventListener('touchmove', handleTouchMove);
canvas.addEventListener('touchend', stopDrawing);

// 按鈕事件
document.querySelector('button[onclick="predict()"]').onclick = () => predict(false);
document.querySelector('button[onclick="clearCanvas()"]').onclick = clearCanvas;
eraserBtn.onclick = toggleEraser;
camToggleBtn.onclick = toggleCamera;
voiceBtn.onclick = toggleVoice;
fileInput.onchange = handleFile;

// 初始化系統
init();
