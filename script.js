/**
 * 🌌 手寫數字辨識系統 - 銀河主題版
 * 完整功能版本 - 修復辨識問題
 */

// --- 全域變數初始化 ---
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
let lastX = 0;
let lastY = 0;
let recognition = null;
let isVoiceActive = false;
let isProcessing = false;

// --- 模型載入函數 ---
async function loadModel() {
    try {
        console.log("🌌 正在載入銀河辨識引擎...");
        confDetails.innerText = "🌌 正在載入銀河辨識引擎...";
        
        // 等待 TensorFlow.js 準備就緒
        await tf.ready();
        console.log("TensorFlow.js 版本:", tf.version.tfjs);
        
        // 載入 TensorFlow.js 模型
        const modelUrl = 'tfjs_model/model.json';
        model = await tf.loadLayersModel(modelUrl);
        
        console.log("✅ 模型載入成功！");
        console.log("模型輸入形狀:", model.inputs[0].shape);
        console.log("模型輸出形狀:", model.outputs[0].shape);
        
        // 模型暖身（使用隨機資料）
        const warmupTensor = tf.randomUniform([1, 28, 28, 1], 0, 1);
        const warmupResult = model.predict(warmupTensor);
        warmupTensor.dispose();
        warmupResult.dispose();
        
        confDetails.innerText = "🚀 系統就緒，請開始書寫數字";
        return true;
    } catch (error) {
        console.error("❌ 模型載入失敗:", error);
        confDetails.innerHTML = `<span style="color: #ff4d4d">❌ 模型載入失敗: ${error.message}</span>`;
        return false;
    }
}

// --- 影像處理函數 (移植自 p.py) ---

// 簡易高斯模糊
function simpleGaussianBlur(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const result = new ImageData(width, height);
    
    // 簡化的 3x3 高斯核
    const kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ];
    const kernelSum = 16;
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let r = 0, g = 0, b = 0;
            
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = ((y + ky) * width + (x + kx)) * 4;
                    const weight = kernel[ky + 1][kx + 1];
                    
                    r += data[idx] * weight;
                    g += data[idx + 1] * weight;
                    b += data[idx + 2] * weight;
                }
            }
            
            const resultIdx = (y * width + x) * 4;
            result.data[resultIdx] = Math.min(255, Math.max(0, r / kernelSum));
            result.data[resultIdx + 1] = Math.min(255, Math.max(0, g / kernelSum));
            result.data[resultIdx + 2] = Math.min(255, Math.max(0, b / kernelSum));
            result.data[resultIdx + 3] = 255;
        }
    }
    
    return result;
}

// Otsu 閾值計算
function calculateOtsuThreshold(imageData) {
    const data = imageData.data;
    const histogram = new Array(256).fill(0);
    
    // 計算灰階直方圖
    for (let i = 0; i < data.length; i += 4) {
        const gray = Math.floor((data[i] + data[i + 1] + data[i + 2]) / 3);
        histogram[gray]++;
    }
    
    // Otsu 算法
    const totalPixels = data.length / 4;
    let sum = 0;
    for (let i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }
    
    let sumBackground = 0;
    let weightBackground = 0;
    let weightForeground = 0;
    let maxVariance = 0;
    let threshold = 0;
    
    for (let i = 0; i < 256; i++) {
        weightBackground += histogram[i];
        if (weightBackground === 0) continue;
        
        weightForeground = totalPixels - weightBackground;
        if (weightForeground === 0) break;
        
        sumBackground += i * histogram[i];
        
        const meanBackground = sumBackground / weightBackground;
        const meanForeground = (sum - sumBackground) / weightForeground;
        
        const variance = weightBackground * weightForeground * 
                         Math.pow(meanBackground - meanForeground, 2);
        
        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = i;
        }
    }
    
    return threshold;
}

// 二值化影像
function binarizeImage(imageData, threshold) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const binaryData = new Uint8Array(width * height);
    
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
        binaryData[j] = gray > threshold ? 255 : 0;
    }
    
    return { data: binaryData, width, height };
}

// 連通域分析
function findConnectedComponents(binaryImage) {
    const { data, width, height } = binaryImage;
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
            
            if (!visited[idx] && data[idx] === 255) {
                // BFS 搜尋連通域
                const queue = [[x, y]];
                visited[idx] = true;
                
                let minX = x, maxX = x, minY = y, maxY = y;
                let pixelCount = 0;
                
                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    const cIdx = cy * width + cx;
                    
                    pixelCount++;
                    
                    minX = Math.min(minX, cx);
                    maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy);
                    maxY = Math.max(maxY, cy);
                    
                    // 檢查8鄰居
                    for (const [dx, dy] of directions) {
                        const nx = cx + dx;
                        const ny = cy + dy;
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            
                            if (!visited[nIdx] && data[nIdx] === 255) {
                                visited[nIdx] = true;
                                queue.push([nx, ny]);
                            }
                        }
                    }
                }
                
                const w = maxX - minX + 1;
                const h = maxY - minY + 1;
                const aspectRatio = w / h;
                const area = pixelCount;
                const solidity = area / (w * h);
                
                components.push({
                    x: minX,
                    y: minY,
                    w: w,
                    h: h,
                    area: area,
                    aspectRatio: aspectRatio,
                    solidity: solidity
                });
            }
        }
    }
    
    return components;
}

// 進階預處理 (對應 p.py 中的 advanced_preprocess)
function advancedPreprocess(roiImage) {
    const { data, width, height } = roiImage;
    
    // 1. 膨脹操作 (簡化版本)
    const dilatedData = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            let maxVal = 0;
            
            // 2x2 核膨脹
            for (let dy = 0; dy <= 1; dy++) {
                for (let dx = 0; dx <= 1; dx++) {
                    const nx = x + dx;
                    const ny = y + dy;
                    if (nx < width && ny < height) {
                        const nIdx = ny * width + nx;
                        maxVal = Math.max(maxVal, data[nIdx]);
                    }
                }
            }
            
            dilatedData[idx] = maxVal;
        }
    }
    
    // 2. 動態 Padding
    const padding = Math.floor(Math.max(height, width) * 0.45);
    const paddedWidth = width + 2 * padding;
    const paddedHeight = height + 2 * padding;
    
    const paddedData = new Uint8Array(paddedWidth * paddedHeight);
    
    // 填充黑色背景
    for (let i = 0; i < paddedData.length; i++) {
        paddedData[i] = 0;
    }
    
    // 複製膨脹後的影像到中央
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const srcIdx = y * width + x;
            const dstIdx = (y + padding) * paddedWidth + (x + padding);
            paddedData[dstIdx] = dilatedData[srcIdx];
        }
    }
    
    // 3. 縮放至 28x28
    const targetSize = 28;
    const scaledData = new Uint8Array(targetSize * targetSize);
    
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
    
    // 4. 質心校正
    let sumX = 0, sumY = 0, total = 0;
    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const idx = y * targetSize + x;
            if (scaledData[idx] > 128) {
                sumX += x;
                sumY += y;
                total++;
            }
        }
    }
    
    let cx = 14, cy = 14;
    if (total > 0) {
        cx = sumX / total;
        cy = sumY / total;
    }
    
    const dx = 14 - cx;
    const dy = 14 - cy;
    
    const correctedData = new Uint8Array(targetSize * targetSize);
    
    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const srcX = Math.round(x - dx);
            const srcY = Math.round(y - dy);
            
            if (srcX >= 0 && srcX < targetSize && srcY >= 0 && srcY < targetSize) {
                const srcIdx = srcY * targetSize + srcX;
                correctedData[y * targetSize + x] = scaledData[srcIdx];
            } else {
                correctedData[y * targetSize + x] = 0;
            }
        }
    }
    
    return correctedData;
}

// --- 主辨識函數 ---
async function predictManual() {
    return await predict(false);
}

async function predict(isRealtime = false) {
    // 防止重複處理
    if (isProcessing) return;
    isProcessing = true;
    
    // 檢查模型
    if (!model) {
        const loaded = await loadModel();
        if (!loaded) {
            digitDisplay.innerText = "❌";
            confDetails.innerHTML = "<b>錯誤：</b>模型未載入";
            isProcessing = false;
            return;
        }
    }
    
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
        
        const imageData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
        
        // 檢查影像是否為空
        let isEmpty = true;
        for (let i = 0; i < imageData.data.length; i += 4) {
            const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
            if (gray > 10 && gray < 245) {
                isEmpty = false;
                break;
            }
        }
        
        if (isEmpty) {
            digitDisplay.innerText = "---";
            confDetails.innerText = "請在畫布上書寫數字";
            isProcessing = false;
            return;
        }
        
        // 1. 計算平均亮度
        let totalBrightness = 0;
        for (let i = 0; i < imageData.data.length; i += 4) {
            const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
            totalBrightness += gray;
        }
        const avgBrightness = totalBrightness / (imageData.data.length / 4);
        
        // 2. 轉為灰階並可能反轉
        const grayImageData = new ImageData(canvas.width, canvas.height);
        for (let i = 0; i < imageData.data.length; i += 4) {
            let gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
            
            // 背景反轉檢測
            if (avgBrightness > 120) {
                gray = 255 - gray;
            }
            
            grayImageData.data[i] = gray;
            grayImageData.data[i + 1] = gray;
            grayImageData.data[i + 2] = gray;
            grayImageData.data[i + 3] = 255;
        }
        
        // 3. 高斯模糊
        const blurred = simpleGaussianBlur(grayImageData);
        
        // 4. Otsu 二值化
        const threshold = calculateOtsuThreshold(blurred);
        const binaryImage = binarizeImage(blurred, threshold);
        
        // 5. 連通域分析
        const components = findConnectedComponents(binaryImage);
        
        // 6. 過濾連通域
        const MIN_AREA = isRealtime ? 500 : 150;
        const filteredComponents = [];
        
        for (const comp of components) {
            // 面積過小
            if (comp.area < MIN_AREA) continue;
            
            // 排除過於細長或寬大的線條
            if (comp.aspectRatio > 2.5 || comp.aspectRatio < 0.15) continue;
            
            // Solidity (填滿率) 檢查
            if (comp.solidity < 0.15) continue;
            
            // 邊緣無效區過濾
            const border = 8;
            if (comp.x < border || comp.y < border || 
                (comp.x + comp.w) > (canvas.width - border) || 
                (comp.y + comp.h) > (canvas.height - border)) {
                if (comp.area < 1000) continue;
            }
            
            filteredComponents.push(comp);
        }
        
        // 排序 (由左至右)
        filteredComponents.sort((a, b) => a.x - b.x);
        
        let finalResult = "";
        const details = [];
        const validBoxes = [];
        
        // 7. 對每個區域進行辨識
        for (const comp of filteredComponents) {
            // 提取 ROI 數據
            const roiBinaryData = {
                data: new Uint8Array(comp.w * comp.h),
                width: comp.w,
                height: comp.h
            };
            
            // 從原始二值化影像中提取 ROI
            for (let y = 0; y < comp.h; y++) {
                for (let x = 0; x < comp.w; x++) {
                    const srcX = comp.x + x;
                    const srcY = comp.y + y;
                    const srcIdx = srcY * canvas.width + srcX;
                    const dstIdx = y * comp.w + x;
                    roiBinaryData.data[dstIdx] = binaryImage.data[srcIdx];
                }
            }
            
            // 連體字切割邏輯 (處理寬度大於高度1.3倍的區域)
            if (comp.w > comp.h * 1.3) {
                // 水平投影
                const projection = new Array(comp.w).fill(0);
                for (let x = 0; x < comp.w; x++) {
                    for (let y = 0; y < comp.h; y++) {
                        const idx = y * comp.w + x;
                        if (roiBinaryData.data[idx] === 255) {
                            projection[x]++;
                        }
                    }
                }
                
                // 找到分割點
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
                
                // 分割成兩個子區域
                const subRegions = [
                    { x: 0, width: splitX, height: comp.h },
                    { x: splitX, width: comp.w - splitX, height: comp.h }
                ];
                
                for (const subRegion of subRegions) {
                    if (subRegion.width < 5) continue;
                    
                    // 提取子區域
                    const subData = {
                        data: new Uint8Array(subRegion.width * subRegion.height),
                        width: subRegion.width,
                        height: subRegion.height
                    };
                    
                    for (let y = 0; y < subRegion.height; y++) {
                        for (let x = 0; x < subRegion.width; x++) {
                            const srcX = subRegion.x + x;
                            const srcIdx = y * comp.w + srcX;
                            const dstIdx = y * subRegion.width + x;
                            subData.data[dstIdx] = roiBinaryData.data[srcIdx];
                        }
                    }
                    
                    // 進階預處理
                    const processedData = advancedPreprocess(subData);
                    
                    // 轉換為 Tensor
                    const floatData = new Float32Array(processedData.length);
                    for (let i = 0; i < processedData.length; i++) {
                        floatData[i] = processedData[i] / 255.0;
                    }
                    
                    const tensor = tf.tensor4d(floatData, [1, 28, 28, 1]);
                    
                    // 預測
                    const prediction = model.predict(tensor);
                    const scores = await prediction.data();
                    const digit = prediction.argMax(-1).dataSync()[0];
                    const confidence = Math.max(...scores);
                    
                    tensor.dispose();
                    prediction.dispose();
                    
                    if (confidence > 0.8) {
                        finalResult += digit.toString();
                        details.push({
                            digit: digit,
                            conf: `${(confidence * 100).toFixed(1)}%`
                        });
                    }
                }
                
                continue;
            }
            
            // 一般數字預測
            // 進階預處理
            const processedData = advancedPreprocess(roiBinaryData);
            
            // 轉換為 Tensor
            const floatData = new Float32Array(processedData.length);
            for (let i = 0; i < processedData.length; i++) {
                floatData[i] = processedData[i] / 255.0;
            }
            
            const tensor = tf.tensor4d(floatData, [1, 28, 28, 1]);
            
            // 預測
            const prediction = model.predict(tensor);
            const scores = await prediction.data();
            const digit = prediction.argMax(-1).dataSync()[0];
            const confidence = Math.max(...scores);
            
            tensor.dispose();
            prediction.dispose();
            
            // 信心度過濾
            if (isRealtime && confidence < 0.85) {
                continue;
            }
            
            finalResult += digit.toString();
            details.push({
                digit: digit,
                conf: `${(confidence * 100).toFixed(1)}%`
            });
            
            validBoxes.push({
                x: comp.x,
                y: comp.y,
                w: comp.w,
                h: comp.h
            });
        }
        
        // 8. 更新顯示
        if (finalResult) {
            digitDisplay.innerText = finalResult;
            
            // 添加動畫效果
            digitDisplay.style.transform = "scale(1.2)";
            setTimeout(() => {
                digitDisplay.style.transform = "scale(1)";
            }, 300);
            
            // 視覺回饋
            addVisualFeedback("#2ecc71");
        } else {
            digitDisplay.innerText = "---";
            if (isRealtime) {
                confDetails.innerText = "正在尋找數字...";
            } else {
                confDetails.innerText = "未偵測到有效數字";
            }
        }
        
        updateDetails(details);
        
        // 9. 如果是即時模式，畫出偵測框
        if (isRealtime && cameraStream && validBoxes.length > 0) {
            // 清除畫布（只清除框框區域）
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // 重新繪製框框
            validBoxes.forEach((box, index) => {
                // 畫綠色框框
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.strokeRect(box.x, box.y, box.w, box.h);
                
                // 畫辨識到的數字
                const detectedDigit = details[index] ? details[index].digit : "";
                ctx.fillStyle = "#00FF00";
                ctx.font = "bold 24px Arial";
                ctx.fillText(detectedDigit.toString(), box.x, box.y - 5);
            });
            
            // 恢復畫筆設定
            updatePen();
        }
        
        isProcessing = false;
        return {
            full_digit: finalResult,
            details: details,
            boxes: validBoxes
        };
        
    } catch (error) {
        console.error("辨識錯誤:", error);
        digitDisplay.innerText = "❌";
        confDetails.innerHTML = `<b>錯誤：</b>${error.message}`;
        addVisualFeedback("#e74c3c");
        isProcessing = false;
        return { error: error.message };
    }
}

// --- UI 功能 ---

// 初始化
function init() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();
    
    // 載入模型
    loadModel();
    
    // 初始提示
    digitDisplay.innerText = "---";
    confDetails.innerText = "請在畫布上書寫數字，然後點擊「開始辨識」";
}

// 更新畫筆
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
    confDetails.innerText = "畫布已清空，請重新書寫";
    addVisualFeedback("#2ecc71");
}

// 視覺回饋
function addVisualFeedback(color) {
    const buttons = document.querySelectorAll('.btn-container button');
    buttons.forEach(btn => {
        const originalBoxShadow = btn.style.boxShadow;
        btn.style.boxShadow = `0 0 20px ${color}`;
        
        setTimeout(() => {
            btn.style.boxShadow = originalBoxShadow;
        }, 300);
    });
}

// 相機功能
async function toggleCamera() {
    if (cameraStream) {
        stopCamera();
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    facingMode: "environment", 
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
            video.srcObject = cameraStream;
            video.style.display = "block";
            mainBox.classList.add('cam-active');
            camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 關閉鏡頭';
            
            // 開始即時辨識
            realtimeInterval = setInterval(async () => {
                await predict(true);
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
    init();
    addVisualFeedback("#34495e");
}

// 檔案上傳
function triggerFile() {
    fileInput.click();
    addVisualFeedback("#3498db");
}

function handleFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // 如果相機開啟，先關閉
    if (cameraStream) stopCamera();
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            clearCanvas();
            
            // 計算適當的尺寸
            const maxWidth = canvas.width - 100;
            const maxHeight = canvas.height - 100;
            let width = img.width;
            let height = img.height;
            
            if (width > maxWidth) {
                height = (maxWidth / width) * height;
                width = maxWidth;
            }
            
            if (height > maxHeight) {
                width = (maxHeight / height) * width;
                height = maxHeight;
            }
            
            // 置中繪製
            const x = (canvas.width - width) / 2;
            const y = (canvas.height - height) / 2;
            
            ctx.drawImage(img, x, y, width, height);
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
        html += "未偵測到有效數字";
    } else {
        data.forEach((item, i) => {
            const color = i % 2 === 0 ? "#a3d9ff" : "#ff6b9d";
            html += `數字 ${i + 1}: <b style="color:${color}">${item.digit}</b> (信心度: ${item.conf})<br>`;
        });
    }
    confDetails.innerHTML = html;
}

// --- 語音功能 ---

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
        console.log("語音識別結果:", transcript);
        
        if (transcript.includes('清除') || transcript.includes('清空')) {
            clearCanvas();
        } else if (transcript.includes('開始') || transcript.includes('辨識')) {
            predict(false);
        } else if (transcript.includes('鏡頭') || transcript.includes('相機')) {
            toggleCamera();
        } else if (transcript.includes('橡皮擦')) {
            toggleEraser();
        } else if (/^\d+$/.test(transcript)) {
            digitDisplay.innerText = transcript;
            confDetails.innerHTML = `<b>語音來源：</b><span style="color:#ff6b9d">${transcript}</span>`;
            addVisualFeedback("#ff6b9d");
        } else {
            confDetails.innerHTML = `<b>語音指令：</b><span style="color:#ff6b9d">${transcript}</span>`;
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
            // 請求麥克風權限
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    // 停止音訊串流以避免佔用麥克風
                    stream.getTracks().forEach(track => track.stop());
                    
                    // 啟動語音識別
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

// --- 繪圖事件處理 ---

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
    ctx.lineTo(x, y);
    ctx.stroke();
    
    lastX = x;
    lastY = y;
}

function draw(e) {
    e.preventDefault();
    
    if (!isDrawing) return;
    
    const { x, y } = getCanvasCoordinates(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    lastX = x;
    lastY = y;
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
        if (!cameraStream) {
            setTimeout(() => predict(false), 300);
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

// --- 事件監聽器 ---

// 滑鼠事件
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// 觸控事件
canvas.addEventListener('touchstart', handleTouchStart);
canvas.addEventListener('touchmove', handleTouchMove);
canvas.addEventListener('touchend', stopDrawing);

// 初始化系統
init();
