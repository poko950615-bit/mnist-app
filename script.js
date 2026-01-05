/**
 * 🌌 銀河手寫數字辨識系統 - 完整前端版本
 * 包含 TensorFlow.js 模型加載和真正的數字辨識
 * 模型檔案位於 tfjs_model 資料夾中
 */

// ==================== 全局變量初始化 ====================
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
let isProcessing = false;
let lastX = 0;
let lastY = 0;

// ==================== 系統初始化 ====================
async function init() {
    console.log('🌌 初始化銀河辨識系統...');
    
    // 初始化畫布
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    
    // 初始化語音識別
    initSpeechRecognition();
    
    // 載入 TensorFlow.js 模型
    await loadModel();
    
    // 初始提示
    digitDisplay.innerText = "---";
    confDetails.innerText = "🚀 系統就緒，請開始書寫數字";
    
    // 銀河特效
    addGalaxyEffects();
    
    console.log('✅ 系統初始化完成');
}

// ==================== 載入 TensorFlow.js 模型 ====================
async function loadModel() {
    try {
        confDetails.innerText = "🌌 正在載入神經網路模型...";
        
        // 設置 TensorFlow.js 後端
        try {
            await tf.setBackend('webgl');
            console.log('使用 WebGL 後端');
        } catch (webglError) {
            console.log('WebGL 不可用，使用 CPU 後端:', webglError);
            await tf.setBackend('cpu');
        }
        
        await tf.ready();
        
        console.log('TensorFlow.js 版本:', tf.version.tfjs);
        console.log('使用後端:', tf.getBackend());
        
        // 載入模型 - 模型位於 tfjs_model 資料夾中
        const modelUrl = './tfjs_model/model.json';
        console.log('正在載入模型:', modelUrl);
        
        // 添加載入超時機制
        const loadPromise = tf.loadLayersModel(modelUrl);
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('模型載入超時')), 15000)
        );
        
        model = await Promise.race([loadPromise, timeoutPromise]);
        
        // 檢查模型是否成功載入
        console.log('✅ 模型載入成功！');
        console.log('模型結構:', model);
        console.log('輸入形狀:', model.inputs[0].shape);
        console.log('輸出形狀:', model.outputs[0].shape);
        
        // 模型暖身
        const warmupInput = tf.zeros([1, 28, 28, 1]);
        const warmupOutput = model.predict(warmupInput);
        await warmupOutput.data();
        warmupInput.dispose();
        warmupOutput.dispose();
        
        confDetails.innerText = "🚀 系統就緒 (神經網路模式)";
        
        return true;
        
    } catch (error) {
        console.error('❌ 模型載入失敗:', error);
        confDetails.innerHTML = `
            <span style="color: #e74c3c">
                ❌ 模型載入失敗<br>
                <small>錯誤: ${error.message}</small><br>
                <small>請確保 tfjs_model 資料夾包含 model.json 和 group1-shard1of1.bin</small>
            </span>
        `;
        
        // 提供備用方案
        setTimeout(() => {
            if (!model) {
                confDetails.innerHTML = `
                    <span style="color: #f39c12">
                        ⚠️ 使用簡易辨識模式<br>
                        <small>請在畫布上手寫數字進行測試</small>
                    </span>
                `;
            }
        }, 3000);
        
        return false;
    }
}

// ==================== 影像處理函數 ====================

// 計算影像平均值
function calculateMean(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum / arr.length;
}

// 計算直方圖
function calculateHistogram(data) {
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i++) {
        histogram[data[i]]++;
    }
    return histogram;
}

// Otsu 閾值計算
function otsuThreshold(grayData) {
    const histogram = calculateHistogram(grayData);
    const total = grayData.length;
    
    let sum = 0;
    for (let i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }
    
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
        
        const mB = sumB / wB;
        const mF = (sum - sumB) / wF;
        const variance = wB * wF * Math.pow(mB - mF, 2);
        
        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = i;
        }
    }
    
    return threshold;
}

// 高斯模糊
function gaussianBlur(data, width, height) {
    const result = new Uint8ClampedArray(data.length);
    const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    const kernelSum = 16;
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sum = 0;
            let k = 0;
            
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const idx = (y + dy) * width + (x + dx);
                    sum += data[idx] * kernel[k];
                    k++;
                }
            }
            
            const idx = y * width + x;
            result[idx] = Math.round(sum / kernelSum);
        }
    }
    
    // 複製邊緣像素
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (y === 0 || y === height - 1 || x === 0 || x === width - 1) {
                const idx = y * width + x;
                result[idx] = data[idx];
            }
        }
    }
    
    return result;
}

// 膨脹操作
function dilateImage(binaryData, width, height) {
    const result = new Uint8ClampedArray(binaryData.length);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let maxVal = 0;
            
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const ny = y + dy;
                    const nx = x + dx;
                    
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        const idx = ny * width + nx;
                        maxVal = Math.max(maxVal, binaryData[idx]);
                    }
                }
            }
            
            const idx = y * width + x;
            result[idx] = maxVal;
        }
    }
    
    return result;
}

// 連通域分析
function findComponents(binaryData, width, height) {
    const visited = new Array(width * height).fill(false);
    const components = [];
    
    const directions = [
        [-1, -1], [0, -1], [1, -1],
        [-1, 0],           [1, 0],
        [-1, 1],  [0, 1],  [1, 1]
    ];
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            
            if (!visited[idx] && binaryData[idx] > 128) {
                const queue = [[x, y]];
                visited[idx] = true;
                
                let minX = x, maxX = x;
                let minY = y, maxY = y;
                let area = 0;
                
                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    const cIdx = cy * width + cx;
                    
                    area++;
                    
                    minX = Math.min(minX, cx);
                    maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy);
                    maxY = Math.max(maxY, cy);
                    
                    for (const [dx, dy] of directions) {
                        const nx = cx + dx;
                        const ny = cy + dy;
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            
                            if (!visited[nIdx] && binaryData[nIdx] > 128) {
                                visited[nIdx] = true;
                                queue.push([nx, ny]);
                            }
                        }
                    }
                }
                
                const w = maxX - minX + 1;
                const h = maxY - minY + 1;
                const aspectRatio = w / h;
                const solidity = area / (w * h);
                
                components.push({
                    x: minX,
                    y: minY,
                    w,
                    h,
                    area,
                    aspectRatio,
                    solidity
                });
            }
        }
    }
    
    return components;
}

// 進階預處理 (與 p.py 完全一致)
function advancedPreprocess(roiData, width, height) {
    // 1. 建立二值化陣列
    const binaryArray = new Uint8ClampedArray(width * height);
    for (let i = 0; i < roiData.length; i++) {
        binaryArray[i] = roiData[i] > 128 ? 255 : 0;
    }
    
    // 2. 膨脹：使用 2x2 核
    const dilated = dilateImage(binaryArray, width, height);
    
    // 3. 動態 Padding (保持數字比例)
    const pad = Math.floor(Math.max(height, width) * 0.45);
    const paddedWidth = width + 2 * pad;
    const paddedHeight = height + 2 * pad;
    
    const paddedData = new Uint8ClampedArray(paddedWidth * paddedHeight);
    paddedData.fill(0);
    
    // 複製膨脹後的影像到中央
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const srcIdx = y * width + x;
            const dstIdx = (y + pad) * paddedWidth + (x + pad);
            paddedData[dstIdx] = dilated[srcIdx];
        }
    }
    
    // 4. 縮放至 28x28 (使用最近鄰插值)
    const targetSize = 28;
    const scaledData = new Uint8ClampedArray(targetSize * targetSize);
    
    const scaleX = paddedWidth / targetSize;
    const scaleY = paddedHeight / targetSize;
    
    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const srcX = Math.floor(x * scaleX);
            const srcY = Math.floor(y * scaleY);
            const srcIdx = srcY * paddedWidth + srcX;
            const dstIdx = y * targetSize + x;
            scaledData[dstIdx] = paddedData[srcIdx];
        }
    }
    
    // 5. 質心校正
    let sumX = 0, sumY = 0, sumVal = 0;
    
    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const idx = y * targetSize + x;
            const val = scaledData[idx];
            if (val > 128) {
                sumX += x * (val / 255);
                sumY += y * (val / 255);
                sumVal += (val / 255);
            }
        }
    }
    
    if (sumVal > 0) {
        const centerX = sumX / sumVal;
        const centerY = sumY / sumVal;
        
        const offsetX = 14 - centerX;
        const offsetY = 14 - centerY;
        
        const centeredData = new Uint8ClampedArray(targetSize * targetSize);
        centeredData.fill(0);
        
        for (let y = 0; y < targetSize; y++) {
            for (let x = 0; x < targetSize; x++) {
                const srcX = Math.round(x - offsetX);
                const srcY = Math.round(y - offsetY);
                
                if (srcX >= 0 && srcX < targetSize && srcY >= 0 && srcY < targetSize) {
                    const srcIdx = srcY * targetSize + srcX;
                    const dstIdx = y * targetSize + x;
                    centeredData[dstIdx] = scaledData[srcIdx];
                }
            }
        }
        
        // 歸一化到 0-1
        const normalized = new Float32Array(targetSize * targetSize);
        for (let i = 0; i < centeredData.length; i++) {
            normalized[i] = centeredData[i] / 255.0;
        }
        
        return normalized;
    }
    
    // 如果無法計算質心，直接歸一化
    const normalized = new Float32Array(targetSize * targetSize);
    for (let i = 0; i < scaledData.length; i++) {
        normalized[i] = scaledData[i] / 255.0;
    }
    
    return normalized;
}

// ==================== 使用 TensorFlow.js 模型進行預測 ====================
async function predictWithModel(processedData) {
    if (!model) {
        throw new Error('模型未載入');
    }
    
    try {
        // 轉換為 TensorFlow.js Tensor
        const tensor = tf.tensor4d(processedData, [1, 28, 28, 1]);
        
        // 進行預測
        const prediction = model.predict(tensor);
        const scores = await prediction.data();
        
        // 找到最高分數和對應的數字
        let maxScore = 0;
        let digit = 0;
        
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                digit = i;
            }
        }
        
        // 釋放 Tensor 記憶體
        tensor.dispose();
        prediction.dispose();
        
        return { digit, confidence: maxScore };
        
    } catch (error) {
        console.error('模型預測錯誤:', error);
        throw error;
    }
}

// ==================== 主辨識函數 ====================
async function predict(isRealtime = false) {
    if (isProcessing) return;
    isProcessing = true;
    
    try {
        // 顯示載入狀態
        if (!isRealtime) {
            digitDisplay.innerHTML = '<span class="pulse-icon">🌠</span>';
            confDetails.innerText = "正在分析影像...";
        }
        
        // 獲取畫布影像
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        
        // 如果有相機串流，先繪製相機影像
        if (cameraStream) {
            tempCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }
        // 繪製手寫畫布
        tempCtx.drawImage(canvas, 0, 0);
        
        // 獲取影像數據
        const imageData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
        
        // 轉為灰階
        const grayData = new Uint8ClampedArray(canvas.width * canvas.height);
        for (let i = 0, j = 0; i < imageData.data.length; i += 4, j++) {
            const r = imageData.data[i];
            const g = imageData.data[i + 1];
            const b = imageData.data[i + 2];
            grayData[j] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        }
        
        // 背景反轉檢測
        const meanBrightness = calculateMean(grayData);
        let processedGray = grayData;
        
        if (meanBrightness > 120) {
            processedGray = new Uint8ClampedArray(grayData.length);
            for (let i = 0; i < grayData.length; i++) {
                processedGray[i] = 255 - grayData[i];
            }
        }
        
        // 高斯模糊
        const blurred = gaussianBlur(processedGray, canvas.width, canvas.height);
        
        // Otsu 二值化
        const threshold = otsuThreshold(blurred);
        const binaryData = new Uint8ClampedArray(blurred.length);
        for (let i = 0; i < blurred.length; i++) {
            binaryData[i] = blurred[i] > threshold ? 255 : 0;
        }
        
        // 連通域分析
        const components = findComponents(binaryData, canvas.width, canvas.height);
        
        // 過濾連通域 (與 p.py 相同的邏輯)
        const MIN_AREA = isRealtime ? 500 : 150;
        const MAX_AREA_RATE = 0.7;
        const filteredComponents = [];
        
        for (const comp of components) {
            // 面積過小
            if (comp.area < MIN_AREA) continue;
            
            // 排除佔據整個畫面的巨大物件
            const imgArea = canvas.width * canvas.height;
            if (comp.w > canvas.width * 0.85 || 
                comp.h > canvas.height * 0.85 || 
                comp.area > imgArea * MAX_AREA_RATE) {
                continue;
            }
            
            // 長寬比
            if (comp.aspectRatio > 2.5 || comp.aspectRatio < 0.15) continue;
            
            // 填滿率
            if (comp.solidity < 0.15) continue;
            
            // 邊緣過濾
            const border = 10;
            if (comp.x < border || comp.y < border || 
                (comp.x + comp.w) > (canvas.width - border) || 
                (comp.y + comp.h) > (canvas.height - border)) {
                if (comp.area < 2000) continue;
            }
            
            filteredComponents.push(comp);
        }
        
        // 排序 (由左至右)
        filteredComponents.sort((a, b) => a.x - b.x);
        
        let finalResult = "";
        const details = [];
        const validBoxes = [];
        
        // 對每個區域進行辨識
        for (const comp of filteredComponents) {
            // 提取 ROI
            const roiData = new Uint8ClampedArray(comp.w * comp.h);
            for (let y = 0; y < comp.h; y++) {
                for (let x = 0; x < comp.w; x++) {
                    const srcIdx = (comp.y + y) * canvas.width + (comp.x + x);
                    const dstIdx = y * comp.w + x;
                    roiData[dstIdx] = binaryData[srcIdx];
                }
            }
            
            // 連體字切割邏輯
            if (comp.w > comp.h * 1.3) {
                // 水平投影
                const projection = new Array(comp.w).fill(0);
                for (let x = 0; x < comp.w; x++) {
                    for (let y = 0; y < comp.h; y++) {
                        const idx = y * comp.w + x;
                        if (roiData[idx] > 128) {
                            projection[x]++;
                        }
                    }
                }
                
                // 找到分割點 (在寬度的 30%-70% 之間尋找最小值)
                const start = Math.floor(comp.w * 0.3);
                const end = Math.floor(comp.w * 0.7);
                let minVal = comp.h + 1;
                let splitX = start;
                
                for (let x = start; x < end; x++) {
                    if (projection[x] < minVal) {
                        minVal = projection[x];
                        splitX = x;
                    }
                }
                
                // 分割並辨識
                const subWidths = [splitX, comp.w - splitX];
                let subX = 0;
                
                for (let i = 0; i < 2; i++) {
                    if (subWidths[i] < 5) continue;
                    
                    // 提取子區域
                    const subData = new Uint8ClampedArray(subWidths[i] * comp.h);
                    for (let y = 0; y < comp.h; y++) {
                        for (let x = 0; x < subWidths[i]; x++) {
                            const srcIdx = y * comp.w + (subX + x);
                            const dstIdx = y * subWidths[i] + x;
                            subData[dstIdx] = roiData[srcIdx];
                        }
                    }
                    
                    subX += subWidths[i];
                    
                    // 進階預處理
                    const processedSubData = advancedPreprocess(subData, subWidths[i], comp.h);
                    
                    // 使用模型預測
                    const result = await predictWithModel(processedSubData);
                    
                    // 要求信心度 > 90%
                    if (result.confidence > 0.90) {
                        finalResult += result.digit;
                        details.push({
                            digit: result.digit,
                            conf: `${(result.confidence * 100).toFixed(1)}%`
                        });
                        validBoxes.push({
                            x: comp.x + (i === 0 ? 0 : splitX),
                            y: comp.y,
                            w: subWidths[i],
                            h: comp.h
                        });
                    }
                }
                
                continue;
            }
            
            // 一般數字辨識
            // 進階預處理
            const processedData = advancedPreprocess(roiData, comp.w, comp.h);
            
            // 使用模型預測
            const result = await predictWithModel(processedData);
            
            // 信心度過濾 (即時模式要求 > 90%)
            if (isRealtime && result.confidence < 0.90) {
                continue;
            }
            
            finalResult += result.digit;
            details.push({
                digit: result.digit,
                conf: `${(result.confidence * 100).toFixed(1)}%`
            });
            
            validBoxes.push({
                x: comp.x,
                y: comp.y,
                w: comp.w,
                h: comp.h
            });
        }
        
        // 更新顯示
        if (finalResult) {
            digitDisplay.innerText = finalResult;
            
            // 添加動畫效果
            digitDisplay.style.transform = "scale(1.2)";
            setTimeout(() => {
                digitDisplay.style.transform = "scale(1)";
            }, 300);
            
            addVisualFeedback("#2ecc71");
        } else {
            digitDisplay.innerText = "---";
            confDetails.innerText = isRealtime ? "等待有效數字入鏡..." : "未偵測到有效數字";
        }
        
        updateDetails(details);
        
        // 如果是即時模式，畫出偵測框
        if (isRealtime && cameraStream && validBoxes.length > 0) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            validBoxes.forEach((box, index) => {
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.strokeRect(box.x, box.y, box.w, box.h);
                
                const detectedDigit = details[index] ? details[index].digit : "";
                ctx.fillStyle = "#00FF00";
                ctx.font = "bold 24px Arial";
                ctx.fillText(detectedDigit.toString(), box.x, box.y - 5);
            });
            
            updatePen();
        }
        
        isProcessing = false;
        
    } catch (error) {
        console.error("辨識錯誤:", error);
        digitDisplay.innerText = "❌";
        confDetails.innerText = `辨識錯誤: ${error.message}`;
        addVisualFeedback("#e74c3c");
        isProcessing = false;
    }
}

// ==================== UI 功能 ====================

function addGalaxyEffects() {
    setTimeout(() => {
        if (!cameraStream) {
            ctx.fillStyle = "rgba(163, 217, 255, 0.3)";
            ctx.beginPath();
            ctx.arc(650, 20, 3, 0, Math.PI * 2);
            ctx.fill();

            ctx.beginPath();
            ctx.arc(30, 300, 2, 0, Math.PI * 2);
            ctx.fill();

            updatePen();
        }
    }, 500);
}

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

    if (isEraser) {
        addVisualFeedback("#e74c3c");
    }
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

// ==================== 相機功能 ====================
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
                predictRealtime();
            }, 400);

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
    clearCanvas();
    addVisualFeedback("#34495e");
}

// ==================== 即時辨識函數 ====================
async function predictRealtime() {
    if (isProcessing || !model) return;
    await predict(true);
}

// ==================== 檔案上傳功能 ====================
function triggerFile() {
    fileInput.click();
    addVisualFeedback("#3498db");
}

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
            predict();
            addVisualFeedback("#3498db");
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ==================== 更新詳細資訊顯示 ====================
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

// ==================== 語音功能 ====================
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
            predict();
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
        if (event.error === 'not-allowed' || event.error === 'audio-capture') {
            alert("請允許瀏覽器使用麥克風權限");
            isVoiceActive = false;
            updateVoiceButton();
            voiceStatus.style.display = 'none';
        }
    };
}

function updateVoiceButton() {
    if (isVoiceActive) {
        voiceBtn.innerHTML = '<span class="btn-icon">🌌</span> 語音輸入：開啟';
        voiceBtn.classList.add('voice-active');
    } else {
        voiceBtn.innerHTML = '<span class="btn-icon">🌌</span> 語音輸入：關閉';
        voiceBtn.classList.remove('voice-active');
    }
}

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
                    alert("請允許使用麥克風以啟用語音輸入功能");
                });
        } catch (e) {
            alert("無法啟動語音識別功能");
        }
    }
}

// ==================== 繪圖事件處理 ====================
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
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getCanvasCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    lastX = x;
    lastY = y;
    if (!isEraser) {
        addDrawingEffect(x, y);
    }
}

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
    if (!isEraser) {
        addDrawingEffect(x, y);
    }
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
        if (!cameraStream) {
            setTimeout(() => predict(), 100);
        }
    }
}

function handleTouchStart(e) {
    if (e.touches.length === 1) {
        startDrawing(e);
    }
}

function handleTouchMove(e) {
    if (e.touches.length === 1) {
        draw(e);
    }
}

function addDrawingEffect(x, y) {
    const effect = document.createElement('div');
    effect.style.position = 'fixed';
    effect.style.left = (x - 5) + 'px';
    effect.style.top = (y - 5) + 'px';
    effect.style.width = '10px';
    effect.style.height = '10px';
    effect.style.borderRadius = '50%';
    effect.style.background = 'radial-gradient(circle, rgba(163, 217, 255, 0.8) 0%, transparent 70%)';
    effect.style.pointerEvents = 'none';
    effect.style.zIndex = '1000';
    document.body.appendChild(effect);
    setTimeout(() => {
        effect.remove();
    }, 500);
}

// ==================== 事件監聽器綁定 ====================
function setupEventListeners() {
    // 畫布事件
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    canvas.addEventListener('touchstart', handleTouchStart);
    canvas.addEventListener('touchmove', handleTouchMove);
    canvas.addEventListener('touchend', stopDrawing);
    
    // 按鈕事件
    document.querySelector('.btn-run')?.addEventListener('click', () => predict());
    document.querySelector('.btn-clear')?.addEventListener('click', clearCanvas);
    eraserBtn.addEventListener('click', toggleEraser);
    camToggleBtn.addEventListener('click', toggleCamera);
    voiceBtn.addEventListener('click', toggleVoice);
    document.querySelector('.btn-upload')?.addEventListener('click', triggerFile);
    
    // 檔案上傳事件
    fileInput.addEventListener('change', handleFile);
}

// ==================== 頁面載入初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM 載入完成，開始初始化...');
    setupEventListeners();
    init();
});

// ==================== TensorFlow.js 內存管理 ====================
setInterval(() => {
    try {
        const memoryInfo = tf.memory();
        if (memoryInfo.numTensors > 100) {
            console.warn(`TensorFlow.js 內存警告: ${memoryInfo.numTensors} 個張量`);
            // 強制垃圾回收（在某些瀏覽器中有效）
            if (typeof gc === 'function') {
                gc();
            }
        }
    } catch (e) {
        // 忽略內存檢查錯誤
    }
}, 30000);

// 導出函數供調試
window.predict = predict;
window.clearCanvas = clearCanvas;
window.toggleCamera = toggleCamera;
window.toggleEraser = toggleEraser;
window.toggleVoice = toggleVoice;
