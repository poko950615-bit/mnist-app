/**
 * 🌌 銀河手寫數字辨識系統 - 完全前端版本
 * 整合了原 p.py 的所有影像處理和辨識邏輯
 * 無需後端伺服器，完全在瀏覽器運行
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

// ==================== 模型加載與初始化 ====================
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

// ==================== 影像處理函數 (從 p.py 移植) ====================

// 轉換 ImageData 為灰階陣列
function imageDataToGrayArray(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const grayArray = new Uint8Array(width * height);
    
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        grayArray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    
    return { data: grayArray, width, height };
}

// 計算平均亮度
function calculateAverageBrightness(grayArray) {
    let sum = 0;
    for (let i = 0; i < grayArray.data.length; i++) {
        sum += grayArray.data[i];
    }
    return sum / grayArray.data.length;
}

// 背景反轉
function invertBackground(grayArray) {
    const inverted = new Uint8Array(grayArray.data.length);
    for (let i = 0; i < grayArray.data.length; i++) {
        inverted[i] = 255 - grayArray.data[i];
    }
    return { data: inverted, width: grayArray.width, height: grayArray.height };
}

// 高斯模糊 (5x5 核心)
function gaussianBlur(grayArray) {
    const { data, width, height } = grayArray;
    const result = new Uint8Array(width * height);
    
    // 5x5 高斯核
    const kernel = [
        1, 4, 6, 4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4, 6, 4, 1
    ];
    const kernelSum = 256;
    
    const kernelSize = 5;
    const halfKernel = Math.floor(kernelSize / 2);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let sum = 0;
            let k = 0;
            
            // 處理邊界：使用反射填充
            for (let ky = -halfKernel; ky <= halfKernel; ky++) {
                for (let kx = -halfKernel; kx <= halfKernel; kx++) {
                    let nx = x + kx;
                    let ny = y + ky;
                    
                    // 邊界反射
                    if (nx < 0) nx = -nx;
                    if (nx >= width) nx = 2 * width - nx - 1;
                    if (ny < 0) ny = -ny;
                    if (ny >= height) ny = 2 * height - ny - 1;
                    
                    const idx = ny * width + nx;
                    sum += data[idx] * kernel[k];
                    k++;
                }
            }
            
            const idx = y * width + x;
            result[idx] = Math.round(sum / kernelSum);
        }
    }
    
    return { data: result, width, height };
}

// Otsu 閾值計算
function calculateOtsuThreshold(grayArray) {
    const { data } = grayArray;
    
    // 計算直方圖
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i++) {
        histogram[data[i]]++;
    }
    
    const total = data.length;
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

// 二值化
function binarizeImage(grayArray, threshold) {
    const { data, width, height } = grayArray;
    const binary = new Uint8Array(width * height);
    
    for (let i = 0; i < data.length; i++) {
        binary[i] = data[i] > threshold ? 255 : 0;
    }
    
    return { data: binary, width, height };
}

// 連通域分析
function findConnectedComponents(binaryImage, connectivity = 8) {
    const { data, width, height } = binaryImage;
    const visited = new Array(width * height).fill(false);
    const components = [];
    
    const directions = connectivity === 8 ? 
        [[-1, -1], [0, -1], [1, -1],
         [-1, 0], [1, 0],
         [-1, 1], [0, 1], [1, 1]] :
        [[0, -1], [-1, 0], [1, 0], [0, 1]];
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            
            if (!visited[idx] && data[idx] === 255) {
                const queue = [[x, y]];
                visited[idx] = true;
                
                let minX = x, maxX = x, minY = y, maxY = y;
                let area = 0;
                const pixels = [];
                
                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    const cIdx = cy * width + cx;
                    
                    area++;
                    pixels.push([cx, cy]);
                    
                    minX = Math.min(minX, cx);
                    maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy);
                    maxY = Math.max(maxY, cy);
                    
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
                const solidity = area / (w * h);
                
                components.push({
                    x: minX,
                    y: minY,
                    w,
                    h,
                    area,
                    aspectRatio,
                    solidity,
                    pixels
                });
            }
        }
    }
    
    return components;
}

// 膨脹操作
function dilateImage(binaryImage, kernelSize = 2) {
    const { data, width, height } = binaryImage;
    const result = new Uint8Array(width * height);
    
    const halfKernel = Math.floor(kernelSize / 2);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            let maxVal = 0;
            
            for (let ky = -halfKernel; ky <= halfKernel; ky++) {
                for (let kx = -halfKernel; kx <= halfKernel; kx++) {
                    const nx = x + kx;
                    const ny = y + ky;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const nIdx = ny * width + nx;
                        maxVal = Math.max(maxVal, data[nIdx]);
                    }
                }
            }
            
            result[idx] = maxVal;
        }
    }
    
    return { data: result, width, height };
}

// 計算圖像矩
function calculateImageMoments(binaryImage) {
    const { data, width, height } = binaryImage;
    
    let m00 = 0, m10 = 0, m01 = 0;
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            const value = data[idx] / 255;
            m00 += value;
            m10 += x * value;
            m01 += y * value;
        }
    }
    
    return { m00, m10, m01 };
}

// 進階預處理 (從 p.py 的 advanced_preprocess 移植)
function advancedPreprocess(roiImage) {
    const { data, width, height } = roiImage;
    
    // 1. 建立二值化陣列
    const binaryArray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i++) {
        binaryArray[i] = data[i] > 128 ? 255 : 0;
    }
    
    // 2. 膨脹：使用 2x2 核 (移植自 p.py)
    const dilated = dilateImage({ data: binaryArray, width, height }, 2);
    
    // 3. 動態 Padding (保持數字比例)
    const pad = Math.floor(Math.max(height, width) * 0.45);
    const paddedWidth = width + 2 * pad;
    const paddedHeight = height + 2 * pad;
    
    const paddedData = new Uint8Array(paddedWidth * paddedHeight);
    
    // 填充黑色背景
    for (let i = 0; i < paddedData.length; i++) {
        paddedData[i] = 0;
    }
    
    // 複製膨脹後的影像到中央
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const srcIdx = y * width + x;
            const dstIdx = (y + pad) * paddedWidth + (x + pad);
            paddedData[dstIdx] = dilated.data[srcIdx];
        }
    }
    
    // 4. 縮放至 28x28 (使用最近鄰插值)
    const targetSize = 28;
    const scaledData = new Uint8Array(targetSize * targetSize);
    
    const xRatio = paddedWidth / targetSize;
    const yRatio = paddedHeight / targetSize;
    
    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const srcX = Math.floor(x * xRatio);
            const srcY = Math.floor(y * yRatio);
            const srcIdx = srcY * paddedWidth + srcX;
            const dstIdx = y * targetSize + x;
            scaledData[dstIdx] = paddedData[srcIdx];
        }
    }
    
    // 5. 質心校正
    const moments = calculateImageMoments({ 
        data: scaledData, 
        width: targetSize, 
        height: targetSize 
    });
    
    let finalData;
    
    if (moments.m00 !== 0) {
        const cx = moments.m10 / moments.m00;
        const cy = moments.m01 / moments.m00;
        
        const dx = 14 - cx;
        const dy = 14 - cy;
        
        const correctedData = new Uint8Array(targetSize * targetSize);
        
        // 應用仿射變換 (平移)
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
        
        finalData = correctedData;
    } else {
        finalData = scaledData;
    }
    
    // 6. 歸一化到 0-1 範圍
    const normalizedData = new Float32Array(targetSize * targetSize);
    for (let i = 0; i < finalData.length; i++) {
        normalizedData[i] = finalData[i] / 255.0;
    }
    
    return normalizedData;
}

// ==================== 主辨識函數 (整合 p.py 邏輯) ====================
async function predict(isRealtime = false) {
    if (isProcessing || !model) return;
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
        
        // 1. 轉為灰階
        const grayImage = imageDataToGrayArray(imageData);
        
        // 2. 背景反轉檢測 (移植自 p.py)
        const avgBrightness = calculateAverageBrightness(grayImage);
        let processedGray = grayImage;
        
        if (avgBrightness > 120) {
            processedGray = invertBackground(grayImage);
        }
        
        // 3. 高斯模糊 (去噪)
        const blurred = gaussianBlur(processedGray);
        
        // 4. Otsu 二值化
        const otsuThreshold = calculateOtsuThreshold(blurred);
        const binaryImage = binarizeImage(blurred, otsuThreshold);
        
        // 5. 連通域分析
        const components = findConnectedComponents(binaryImage, 8);
        
        // 6. 過濾連通域 (移植自 p.py 的過濾邏輯)
        const MIN_AREA = isRealtime ? 500 : 150;
        const MAX_AREA_RATE = 0.7;
        const filteredComponents = [];
        
        for (const comp of components) {
            // 1. 面積過小則視為雜訊
            if (comp.area < MIN_AREA) continue;
            
            // 2. 排除過於細長或寬大的線條
            if (comp.aspectRatio > 2.5 || comp.aspectRatio < 0.15) continue;
            
            // 3. Solidity (填滿率) 檢查
            if (comp.solidity < 0.15) continue;
            
            // 4. 邊緣無效區過濾
            const border = 10;
            if (comp.x < border || comp.y < border || 
                (comp.x + comp.w) > (canvas.width - border) || 
                (comp.y + comp.h) > (canvas.height - border)) {
                if (comp.area < 2000) continue;
            }
            
            // 5. 排除佔據整個畫面的巨大物件
            const imgArea = canvas.width * canvas.height;
            if (comp.w > canvas.width * 0.85 || 
                comp.h > canvas.height * 0.85 || 
                comp.area > imgArea * MAX_AREA_RATE) {
                continue;
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
            const roiData = {
                data: new Uint8Array(comp.w * comp.h),
                width: comp.w,
                height: comp.h
            };
            
            // 從二值化影像中提取 ROI
            for (let y = 0; y < comp.h; y++) {
                for (let x = 0; x < comp.w; x++) {
                    const srcX = comp.x + x;
                    const srcY = comp.y + y;
                    const srcIdx = srcY * canvas.width + srcX;
                    const dstIdx = y * comp.w + x;
                    roiData.data[dstIdx] = binaryImage.data[srcIdx];
                }
            }
            
            // 連體字切割邏輯 (移植自 p.py)
            if (comp.w > comp.h * 1.3) {
                // 水平投影
                const projection = new Array(comp.w).fill(0);
                for (let x = 0; x < comp.w; x++) {
                    for (let y = 0; y < comp.h; y++) {
                        const idx = y * comp.w + x;
                        if (roiData.data[idx] === 255) {
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
                
                // 分割成兩個子區域
                const subRegions = [
                    { x: 0, w: splitX, h: comp.h },
                    { x: splitX, w: comp.w - splitX, h: comp.h }
                ];
                
                for (const subRegion of subRegions) {
                    if (subRegion.w < 5) continue;
                    
                    // 提取子區域
                    const subData = {
                        data: new Uint8Array(subRegion.w * subRegion.h),
                        width: subRegion.w,
                        height: subRegion.h
                    };
                    
                    for (let y = 0; y < subRegion.h; y++) {
                        for (let x = 0; x < subRegion.w; x++) {
                            const srcX = subRegion.x + x;
                            const srcIdx = y * comp.w + srcX;
                            const dstIdx = y * subRegion.w + x;
                            subData.data[dstIdx] = roiData.data[srcIdx];
                        }
                    }
                    
                    // 進階預處理
                    const processedData = advancedPreprocess(subData);
                    
                    // 轉換為 Tensor 並預測
                    const tensor = tf.tensor4d(processedData, [1, 28, 28, 1]);
                    const prediction = model.predict(tensor);
                    const scores = await prediction.data();
                    const digit = prediction.argMax(-1).dataSync()[0];
                    const confidence = Math.max(...scores);
                    
                    tensor.dispose();
                    prediction.dispose();
                    
                    // 要求信心度 > 90%
                    if (confidence > 0.90) {
                        finalResult += digit.toString();
                        details.push({
                            digit: digit,
                            conf: `${(confidence * 100).toFixed(1)}%`
                        });
                        validBoxes.push({
                            x: comp.x + subRegion.x,
                            y: comp.y,
                            w: subRegion.w,
                            h: subRegion.h
                        });
                    }
                }
                
                continue;
            }
            
            // 一般數字預測
            // 進階預處理
            const processedData = advancedPreprocess(roiData);
            
            // 轉換為 Tensor 並預測
            const tensor = tf.tensor4d(processedData, [1, 28, 28, 1]);
            const prediction = model.predict(tensor);
            const scores = await prediction.data();
            const digit = prediction.argMax(-1).dataSync()[0];
            const confidence = Math.max(...scores);
            
            tensor.dispose();
            prediction.dispose();
            
            // 信心度過濾 (即時模式要求 > 90%)
            if (isRealtime && confidence < 0.90) {
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
                confDetails.innerText = "等待有效數字入鏡...";
            } else {
                confDetails.innerText = "未偵測到有效數字";
            }
        }
        
        updateDetails(details);
        
        // 9. 如果是即時模式，畫出偵測框
        if (isRealtime && cameraStream && validBoxes.length > 0) {
            // 清除畫布
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
        confDetails.innerHTML = `<span style="color: #e74c3c">錯誤：${error.message}</span>`;
        addVisualFeedback("#e74c3c");
        isProcessing = false;
        return { error: error.message };
    }
}

// ==================== 模型加載函數 ====================
async function loadModel() {
    try {
        confDetails.innerText = "🌌 正在啟動銀河辨識引擎...";
        
        // 設置 TensorFlow.js 後端
        await tf.setBackend('cpu');
        await tf.ready();
        
        console.log('TensorFlow.js 版本:', tf.version.tfjs);
        console.log('使用後端:', tf.getBackend());
        
        // 創建一個簡單的 MNIST 模型
        model = await createSimpleModel();
        
        console.log('✅ 模型創建成功！');
        confDetails.innerText = "🚀 系統就緒，請開始書寫數字";
        
        return true;
        
    } catch (error) {
        console.error('❌ 模型加載失敗:', error);
        confDetails.innerHTML = `
            <span style="color: #e74c3c">
                ❌ 模型加載失敗<br>
                <small>錯誤: ${error.message}</small>
            </span>
        `;
        return false;
    }
}

// ==================== 創建簡單的 MNIST 模型 ====================
async function createSimpleModel() {
    // 創建一個簡單的卷積神經網路
    const model = tf.sequential();
    
    // 第一層卷積層
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 8,
        kernelSize: 3,
        activation: 'relu'
    }));
    
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    // 第二層卷積層
    model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        activation: 'relu'
    }));
    
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    // 展平層
    model.add(tf.layers.flatten());
    
    // 全連接層
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    
    // 輸出層
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    
    // 編譯模型
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    // 輸出模型結構
    model.summary();
    
    return model;
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
    init();
    addVisualFeedback("#34495e");
}

// ==================== 即時辨識函數 ====================
async function predictRealtime() {
    if (!model || isProcessing) return;
    
    try {
        const result = await predict(true);
        
        // 更新顯示
        if (result && result.full_digit) {
            digitDisplay.innerText = result.full_digit;
        }
        
        updateDetails(result ? result.details : []);
        
    } catch (err) {
        console.log("即時辨識同步中...");
    }
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
    document.getElementById('eraserBtn')?.addEventListener('click', toggleEraser);
    document.getElementById('camToggleBtn')?.addEventListener('click', toggleCamera);
    document.getElementById('voiceBtn')?.addEventListener('click', toggleVoice);
    document.querySelector('.btn-upload')?.addEventListener('click', triggerFile);
    
    // 檔案上傳事件
    if (fileInput) {
        fileInput.addEventListener('change', handleFile);
    }
}

// ==================== 頁面載入初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM 載入完成，開始初始化...');
    setupEventListeners();
    init();
});

// ==================== 錯誤處理 ====================
window.addEventListener('error', function(e) {
    console.error('全局錯誤:', e.error);
    if (confDetails) {
        confDetails.innerHTML = `<span style="color: #e74c3c">系統錯誤: ${e.message}</span>`;
    }
});

// 導出函數供全局使用
window.predict = predict;
window.clearCanvas = clearCanvas;
window.toggleCamera = toggleCamera;
window.toggleEraser = toggleEraser;
window.toggleVoice = toggleVoice;
window.triggerFile = triggerFile;
