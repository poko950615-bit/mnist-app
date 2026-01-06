/**
 * 🌌 銀河手寫數字辨識系統 - 高信心度鏡頭辨識版
 * 修復了 WebGL 錯誤和語音識別重複啟動問題
 * 鏡頭辨識信心度需 > 93% 才顯示
 * 完全前端運行，無需後端伺服器
 */

// ==================== 全局變量初始化 ====================
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const video = document.getElementById('camera-feed');
const digitDisplay = document.getElementById('digit-display');
const confDetails = document.getElementById('conf-details');
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

// ==================== Keras v3 兼容性修復 ====================
class PatchModelLoader {
    constructor(url) { 
        this.url = url; 
        console.log('PatchModelLoader 初始化，URL:', url);
    }
    
    async load() {
        try {
            console.log('開始加載模型...');
            const loader = tf.io.browserHTTPRequest(this.url);
            const artifacts = await loader.load();
            
            // 修復 InputLayer 形狀
            const traverseAndPatch = (obj) => {
                if (!obj || typeof obj !== 'object') return;
                
                if (obj.class_name === 'InputLayer' && obj.config) {
                    const cfg = obj.config;
                    if (!cfg.batchInputShape && !cfg.batch_input_shape) {
                        console.log('修復 InputLayer 形狀');
                        cfg.batchInputShape = [null, 28, 28, 1];
                    }
                }
                
                if (Array.isArray(obj)) {
                    obj.forEach(item => traverseAndPatch(item));
                } else {
                    Object.keys(obj).forEach(key => traverseAndPatch(obj[key]));
                }
            };
            
            if (artifacts.modelTopology) {
                traverseAndPatch(artifacts.modelTopology);
            }

            // 修復權重名稱
            if (artifacts.weightSpecs) {
                artifacts.weightSpecs.forEach(spec => {
                    if (spec.name.includes('sequential/')) {
                        spec.name = spec.name.replace('sequential/', '');
                    }
                });
            }
            
            console.log('模型加載成功');
            return artifacts;
        } catch (error) {
            console.error('PatchModelLoader 錯誤:', error);
            throw error;
        }
    }
}

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

// ==================== 模型加載 (修復 WebGL 錯誤) ====================
async function loadModel() {
    try {
        confDetails.innerText = "🌌 正在啟動銀河辨識引擎...";
        
        // 更穩健的後端初始化
        const availableBackends = tf.engine().backendNames;
        console.log('可用後端:', availableBackends);
        
        // 優先嘗試 WebGL，如果失敗則自動使用 CPU
        let backendToUse = 'cpu';
        try {
            // 檢查 WebGL 支持
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl') || 
                       canvas.getContext('experimental-webgl');
            if (gl) {
                backendToUse = 'webgl';
            }
        } catch (e) {
            console.log('WebGL 不可用，使用 CPU 後端:', e.message);
        }
        
        // 設置後端
        await tf.setBackend(backendToUse);
        await tf.ready();
        
        console.log('TensorFlow.js 版本:', tf.version.tfjs);
        console.log('最終使用後端:', tf.getBackend());
        
        // 如果使用 CPU，添加性能提示
        if (tf.getBackend() === 'cpu') {
            confDetails.innerHTML = `
                🚀 系統就緒（使用 CPU 模式）<br>
                <small>提示：如需更佳性能，請確保瀏覽器支持 WebGL</small>
            `;
        }
        
        // 載入模型（使用修復器）
        const modelUrl = 'tfjs_model/model.json';
        console.log('從以下位置載入模型:', modelUrl);
        
        model = await tf.loadLayersModel(new PatchModelLoader(modelUrl));
        
        console.log('✅ 模型載入成功！');
        console.log('輸入形狀:', model.inputs[0].shape);
        console.log('輸出形狀:', model.outputs[0].shape);
        
        // 模型暖身
        const testInput = tf.zeros([1, 28, 28, 1]);
        const testOutput = model.predict(testInput);
        await testOutput.data();
        testInput.dispose();
        testOutput.dispose();
        
        if (tf.getBackend() === 'webgl') {
            confDetails.innerText = "🚀 系統就緒（WebGL加速）";
        } else {
            confDetails.innerText = "🚀 系統就緒（CPU模式）";
        }
        
        return true;
        
    } catch (error) {
        console.error('❌ 模型載入失敗:', error);
        confDetails.innerHTML = `
            <span style="color: #ff4d4d">
                ❌ 模型載入失敗<br>
                <small>錯誤: ${error.message}</small>
            </span>
        `;
        return false;
    }
}

// ==================== 影像處理函數 (完整移植自 p.py) ====================

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

// 簡化高斯模糊 (3x3 核心)
function simpleGaussianBlur(grayArray) {
    const { data, width, height } = grayArray;
    const result = new Uint8Array(width * height);
    
    const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    const kernelSum = 16;
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sum = 0;
            let k = 0;
            
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = (y + ky) * width + (x + kx);
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
    
    return { data: result, width, height };
}

// Otsu 閾值計算 (完全移植自 OpenCV 算法)
function calculateOtsuThreshold(grayArray) {
    const { data } = grayArray;
    
    // 計算直方圖
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i++) {
        histogram[data[i]]++;
    }
    
    // 計算總像素數和總和
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
        
        // 計算類間方差
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

// 連通域分析 (8-鄰居)
function findConnectedComponents(binaryImage) {
    const { data, width, height } = binaryImage;
    const visited = new Array(width * height).fill(false);
    const components = [];
    
    // 8方向鄰居
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
                const solidity = area / (w * h);
                
                components.push({
                    x: minX,
                    y: minY,
                    w: w,
                    h: h,
                    area: area,
                    aspectRatio: aspectRatio,
                    solidity: solidity,
                    pixels: pixels
                });
            }
        }
    }
    
    return components;
}

// 膨脹操作 (2x2 核)
function dilateBinary(binaryImage, kernelSize = 2) {
    const { data, width, height } = binaryImage;
    const result = new Uint8Array(width * height);
    
    const half = Math.floor(kernelSize / 2);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            let maxVal = 0;
            
            // 檢查核範圍
            for (let ky = -half; ky <= half; ky++) {
                for (let kx = -half; kx <= half; kx++) {
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

// 計算圖像矩 (用於質心計算)
function calculateImageMoments(binaryImage) {
    const { data, width, height } = binaryImage;
    
    let m00 = 0, m10 = 0, m01 = 0;
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (data[idx] > 0) {
                const value = data[idx] / 255; // 正規化到 0-1
                m00 += value;
                m10 += x * value;
                m01 += y * value;
            }
        }
    }
    
    return { m00, m10, m01 };
}

// 進階預處理 (完全移植自 p.py 的 advanced_preprocess)
function advancedPreprocess(roiImage) {
    const { data, width, height } = roiImage;
    
    // 1. 建立二值化陣列
    const binaryArray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i++) {
        binaryArray[i] = data[i] > 128 ? 255 : 0;
    }
    
    // 2. 膨脹：使用 2x2 核
    const kernelSize = 2;
    const halfKernel = Math.floor(kernelSize / 2);
    const dilated = new Uint8Array(width * height);
    
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
                        maxVal = Math.max(maxVal, binaryArray[nIdx]);
                    }
                }
            }
            
            dilated[idx] = maxVal;
        }
    }
    
    // 3. 動態 Padding
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
            paddedData[dstIdx] = dilated[srcIdx];
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
    const moments = calculateImageMoments({ data: scaledData, width: targetSize, height: targetSize });
    
    if (moments.m00 !== 0) {
        const cx = moments.m10 / moments.m00;
        const cy = moments.m01 / moments.m00;
        
        // 計算平移矩陣
        const dx = 14 - cx;
        const dy = 14 - cy;
        
        const correctedData = new Uint8Array(targetSize * targetSize);
        
        // 應用仿射變換
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
        
        // 6. 歸一化到 0-1 範圍
        const normalizedData = new Float32Array(targetSize * targetSize);
        for (let i = 0; i < correctedData.length; i++) {
            normalizedData[i] = correctedData[i] / 255.0;
        }
        
        return normalizedData;
    } else {
        // 如果 m00 為 0，直接返回縮放後的數據
        const normalizedData = new Float32Array(targetSize * targetSize);
        for (let i = 0; i < scaledData.length; i++) {
            normalizedData[i] = scaledData[i] / 255.0;
        }
        
        return normalizedData;
    }
}

// ==================== 新增：移植自 Python app.py 的鏡頭辨識核心 ====================

// 中值模糊 (移植自 Python 的 cv2.medianBlur)
function medianBlur(grayArray, kernelSize = 5) {
    const { data, width, height } = grayArray;
    const result = new Uint8Array(width * height);
    const radius = Math.floor(kernelSize / 2);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const values = [];
            
            // 收集核內的所有值
            for (let ky = -radius; ky <= radius; ky++) {
                for (let kx = -radius; kx <= radius; kx++) {
                    const nx = x + kx;
                    const ny = y + ky;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const idx = ny * width + nx;
                        values.push(data[idx]);
                    }
                }
            }
            
            // 計算中值
            values.sort((a, b) => a - b);
            const median = values[Math.floor(values.length / 2)];
            result[y * width + x] = median;
        }
    }
    
    return { data: result, width, height };
}

// 自適應閾值 (移植自 Python 的 cv2.adaptiveThreshold)
function adaptiveThreshold(grayArray, blockSize = 31, C = 12) {
    const { data, width, height } = grayArray;
    const result = new Uint8Array(width * height);
    const radius = Math.floor(blockSize / 2);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let sum = 0;
            let count = 0;
            
            // 計算局部平均值
            for (let ky = -radius; ky <= radius; ky++) {
                for (let kx = -radius; kx <= radius; kx++) {
                    const nx = x + kx;
                    const ny = y + ky;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const idx = ny * width + nx;
                        sum += data[idx];
                        count++;
                    }
                }
            }
            
            const mean = sum / count;
            const threshold = mean - C;
            
            // 二值化取反 (THRESH_BINARY_INV)
            result[y * width + x] = data[y * width + x] > threshold ? 0 : 255;
        }
    }
    
    return { data: result, width, height };
}

// Python 風格的 ROI 預處理 (專門為數字1優化)
function pythonStylePreprocess(roiBinary, originalBox) {
    const { data, width, height } = roiBinary;
    
    // 【重要修正：救回數字1的核心邏輯】
    // 不要直接resize，而是先建立一個「正方形黑底」，將數字置中
    // 這樣瘦長的 "1" 才不會被拉成一個充滿格子的正方形
    
    // 1. 找到最大邊長
    const size = Math.max(width, height);
    
    // 2. 增加40%的留白，模仿MNIST數據集 (Python版是0.4)
    const pad = Math.floor(size * 0.4);
    
    // 3. 建立正方形黑底
    const squareSize = size + pad * 2;
    const squareData = new Uint8Array(squareSize * squareSize);
    
    // 全部設為0 (黑色背景)
    for (let i = 0; i < squareData.length; i++) {
        squareData[i] = 0;
    }
    
    // 4. 計算置中偏移
    const offX = (size - width) / 2 + pad;
    const offY = (size - height) / 2 + pad;
    
    // 5. 將ROI複製到正方形中央
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const srcIdx = y * width + x;
            const dstIdx = Math.floor(y + offY) * squareSize + Math.floor(x + offX);
            squareData[dstIdx] = data[srcIdx];
        }
    }
    
    // 6. 縮放到28x28 (使用最近鄰插值)
    const targetSize = 28;
    const scaledData = new Uint8Array(targetSize * targetSize);
    
    const xRatio = squareSize / targetSize;
    const yRatio = squareSize / targetSize;
    
    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
            const srcX = Math.floor(x * xRatio);
            const srcY = Math.floor(y * yRatio);
            const srcIdx = srcY * squareSize + srcX;
            const dstIdx = y * targetSize + x;
            scaledData[dstIdx] = squareData[srcIdx];
        }
    }
    
    // 7. 歸一化到0-1範圍 (使用MNIST的標準化參數)
    const normalizedData = new Float32Array(targetSize * targetSize);
    for (let i = 0; i < scaledData.length; i++) {
        // 使用MNIST標準化: (x/255.0 - 0.1307) / 0.3081
        normalizedData[i] = (scaledData[i] / 255.0 - 0.1307) / 0.3081;
    }
    
    return normalizedData;
}

// ==================== 主辨識函數 (整合Python版鏡頭辨識邏輯) ====================
async function predict(isRealtime = false) {
    // 防止重複處理
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
        
        let finalResult = "";
        const details = [];
        const validBoxes = [];
        
        // =========== 根據模式選擇處理方式 ===========
        if (isRealtime && cameraStream) {
            // =========== 鏡頭辨識模式 (使用Python移植邏輯) ===========
            const width = canvas.width;
            const height = canvas.height;
            
            // 1. 轉為灰階
            const grayImage = imageDataToGrayArray(imageData);
            
            // 2. 中值模糊 (移植自Python的cv2.medianBlur(gray, 5))
            const blurred = medianBlur(grayImage, 5);
            
            // 3. 自適應閾值二值化 (移植自Python的adaptiveThreshold)
            // 使用高斯自適應，區塊大小31，常數12 (Python版參數)
            const binaryImage = adaptiveThreshold(blurred, 31, 12);
            
            // 4. 輕微膨脹：補強數字1的線條連貫性 (Python版是2x2核，迭代1次)
            const dilated = dilateBinary(binaryImage, 2);
            
            // 5. 尋找連通域
            const components = findConnectedComponents(dilated);
            
            // 6. 過濾條件 (移植自Python版的過濾邏輯)
            const foundComponents = [];
            
            // 計算中心熱區
            const hotZoneLeft = 0.2 * width;
            const hotZoneRight = 0.8 * width;
            const hotZoneTop = 0.2 * height;
            const hotZoneBottom = 0.8 * height;
            
            // 計算面積範圍
            const totalPixels = width * height;
            const minArea = totalPixels * 0.002;  // 0.2%
            const maxArea = totalPixels * 0.2;    // 20%
            
            for (const comp of components) {
                // 計算中心點
                const centerX = comp.x + comp.w / 2;
                const centerY = comp.y + comp.h / 2;
                
                // 檢查是否在中心熱區
                const inHotZone = (centerX > hotZoneLeft && centerX < hotZoneRight && 
                                 centerY > hotZoneTop && centerY < hotZoneBottom);
                
                // 檢查面積範圍
                const areaOK = (comp.area > minArea && comp.area < maxArea);
                
                // 檢查寬高比 (Python版放寬到0.05-1.2以捕捉瘦長的1)
                const aspectRatio = comp.w / comp.h;
                const aspectRatioOK = (aspectRatio > 0.05 && aspectRatio < 1.2);
                
                if (inHotZone && areaOK && aspectRatioOK) {
                    foundComponents.push(comp);
                }
            }
            
            // 7. 按x座標排序 (由左到右)
            foundComponents.sort((a, b) => a.x - b.x);
            
            // 8. 對每個區域進行辨識
            for (const comp of foundComponents) {
                // 提取ROI數據
                const roiData = {
                    data: new Uint8Array(comp.w * comp.h),
                    width: comp.w,
                    height: comp.h
                };
                
                // 從二值化影像中提取ROI
                for (let y = 0; y < comp.h; y++) {
                    for (let x = 0; x < comp.w; x++) {
                        const srcX = comp.x + x;
                        const srcY = comp.y + y;
                        const srcIdx = srcY * width + srcX;
                        const dstIdx = y * comp.w + x;
                        roiData.data[dstIdx] = dilated.data[srcIdx];
                    }
                }
                
                // 使用Python風格的預處理 (專門優化數字1)
                const processedData = pythonStylePreprocess(roiData, comp);
                
                // 轉換為Tensor並預測
                const tensor = tf.tensor4d(processedData, [1, 28, 28, 1]);
                const prediction = model.predict(tensor);
                const scores = await prediction.data();
                const digit = prediction.argMax(-1).dataSync()[0];
                const confidence = Math.max(...scores);
                
                tensor.dispose();
                prediction.dispose();
                
                // =========== 修改這裡：信心度過濾從 0.70 提高到 0.93 ===========
                if (confidence > 0.93) {  // 從 0.70 改為 0.93
                    finalResult += digit.toString();
                    details.push({
                        digit: digit,
                        conf: `${(confidence * 100).toFixed(1)}%`,
                        rawConfidence: confidence
                    });
                    
                    validBoxes.push({
                        x: comp.x,
                        y: comp.y,
                        w: comp.w,
                        h: comp.h
                    });
                } else {
                    // 記錄低信心度的偵測
                    console.log(`跳過數字 ${digit}，信心度 ${(confidence*100).toFixed(1)}% < 93%`);
                }
            }
            
            // 9. 畫出中心熱區 (白色邊框)
            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = 1;
            ctx.strokeRect(hotZoneLeft, hotZoneTop, hotZoneRight - hotZoneLeft, hotZoneBottom - hotZoneTop);
            
        } else {
            // =========== 手寫辨識模式 (使用原有邏輯) ===========
            // 1. 轉為灰階
            const grayImage = imageDataToGrayArray(imageData);
            
            // 2. 背景反轉檢測
            const avgBrightness = calculateAverageBrightness(grayImage);
            let processedGray = grayImage;
            
            if (avgBrightness > 120) {
                processedGray = invertBackground(grayImage);
            }
            
            // 3. 高斯模糊 (去噪)
            const blurred = simpleGaussianBlur(processedGray);
            
            // 4. Otsu 二值化
            const otsuThreshold = calculateOtsuThreshold(blurred);
            const binaryImage = binarizeImage(blurred, otsuThreshold);
            
            // 5. 連通域分析
            const components = findConnectedComponents(binaryImage);
            
            // 6. 過濾連通域
            const MIN_AREA = isRealtime ? 500 : 150;
            const filteredComponents = [];
            
            for (const comp of components) {
                // 1. 面積過小則視為雜訊
                if (comp.area < MIN_AREA) continue;
                
                // 2. 排除過於細長或寬大的線條
                if (comp.aspectRatio > 2.5 || comp.aspectRatio < 0.15) continue;
                
                // 3. Solidity (填滿率) 檢查
                if (comp.solidity < 0.15) continue;
                
                // 4. 邊緣無效區過濾
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
                
                // 連體字切割邏輯
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
                const processedData = advancedPreprocess(roiData);
                
                // 轉換為 Tensor 並預測
                const tensor = tf.tensor4d(processedData, [1, 28, 28, 1]);
                const prediction = model.predict(tensor);
                const scores = await prediction.data();
                const digit = prediction.argMax(-1).dataSync()[0];
                const confidence = Math.max(...scores);
                
                tensor.dispose();
                prediction.dispose();
                
                // 信心度過濾 (手寫模式保持 0.8)
                if (confidence > 0.8) {
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
            }
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
            
            // 更新詳細資訊
            updateDetails(details);
            
            if (isRealtime) {
                confDetails.innerHTML = `<span style="color:#2ecc71">✅ 高信心度辨識: ${finalResult} (信心度 > 93%)</span>`;
            } else {
                confDetails.innerHTML = `<span style="color:#2ecc71">✅ 辨識完成: ${finalResult}</span>`;
            }
        } else {
            digitDisplay.innerText = "---";
            if (isRealtime) {
                confDetails.innerText = "等待高信心度數字 (>93%)...";
            } else {
                confDetails.innerText = "未偵測到有效數字";
            }
        }
        
        // 9. 如果是即時模式，畫出偵測框 (只顯示信心度 > 93% 的)
        if (isRealtime && cameraStream && validBoxes.length > 0) {
            // 清除畫布
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // 重新繪製框框 (只繪製信心度 > 93% 的)
            validBoxes.forEach((box, index) => {
                // 畫綠色框框
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.strokeRect(box.x, box.y, box.w, box.h);
                
                // 畫辨識到的數字和信心度
                const detectedDigit = details[index] ? details[index].digit : "";
                const confidence = details[index] ? details[index].conf : "";
                ctx.fillStyle = "#00FF00";
                ctx.font = "bold 24px Arial";
                ctx.fillText(`${detectedDigit} (${confidence})`, box.x, box.y - 5);
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

// ==================== UI 功能 ====================

// 添加銀河主題效果
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

            // 恢復畫筆設置
            updatePen();
        }
    }, 500);
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

// 切換橡皮擦模式
function toggleEraser() {
    isEraser = !isEraser;
    const eraserBtn = document.getElementById('eraserBtn');
    if (eraserBtn) {
        eraserBtn.innerText = isEraser ? "🧽 橡皮擦：開啟" : "🧽 橡皮擦：關閉";
        eraserBtn.classList.toggle('eraser-active', isEraser);
    }
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
    confDetails.innerText = "🪐 畫布已清空，請重新書寫";
    addVisualFeedback("#2ecc71");
    addGalaxyEffects();
}

// 視覺回饋效果
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
        return;
    }
    
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
        document.getElementById('mainBox').classList.add('cam-active');
        
        const camToggleBtn = document.getElementById('camToggleBtn');
        if (camToggleBtn) {
            camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 關閉鏡頭';
        }
        
        // 開始即時辨識
        realtimeInterval = setInterval(async () => {
            await predict(true);
        }, 800); // 降低頻率以減少性能壓力
        
        clearCanvas();
        confDetails.innerText = "📷 鏡頭已開啟，只顯示信心度 > 93% 的數字";
        addVisualFeedback("#9b59b6");
        
    } catch (err) {
        console.error('鏡頭啟動失敗:', err);
        alert("無法啟動鏡頭：請確保已授予相機權限");
    }
}

// 停止相機
function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    if (realtimeInterval) {
        clearInterval(realtimeInterval);
        realtimeInterval = null;
    }
    
    video.style.display = "none";
    document.getElementById('mainBox').classList.remove('cam-active');
    
    const camToggleBtn = document.getElementById('camToggleBtn');
    if (camToggleBtn) {
        camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 開啟鏡頭';
    }
    
    init(); // 重新初始化畫布
    addVisualFeedback("#34495e");
}

// 檔案上傳
function triggerFile() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.click();
    }
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
            const ratio = Math.min(
                canvas.width / img.width * 0.8,
                canvas.height / img.height * 0.8
            );
            const w = img.width * ratio;
            const h = img.height * ratio;
            
            // 置中繪製
            const x = (canvas.width - w) / 2;
            const y = (canvas.height - h) / 2;
            
            ctx.drawImage(img, x, y, w, h);
            predict(false);
            addVisualFeedback("#3498db");
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// 更新詳細資訊顯示
function updateDetails(data) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (!data || data.length === 0) {
        html += "未偵測到高信心度數字 (需 > 93%)";
    } else {
        data.forEach((item, i) => {
            const color = item.rawConfidence > 0.95 ? "#2ecc71" : 
                         item.rawConfidence > 0.93 ? "#f1c40f" : "#ff6b9d";
            html += `數字 ${i + 1}: <b style="color:${color}">${item.digit}</b> (信心度: ${item.conf})<br>`;
        });
    }
    confDetails.innerHTML = html;
}

// ==================== 語音功能 (修復重複啟動錯誤) ====================

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) voiceBtn.style.display = 'none';
        return;
    }
    
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-TW';
    recognition.continuous = true;
    recognition.interimResults = false;
    
    // 添加重試計數器
    let retryCount = 0;
    const MAX_RETRIES = 3;
    
    recognition.onstart = () => {
        isVoiceActive = true;
        retryCount = 0; // 重置重試計數
        updateVoiceButton();
        if (voiceStatus) {
            voiceStatus.style.display = 'block';
            voiceStatus.innerHTML = '<span class="pulse-icon">🎙️</span> 語音辨識已啟動';
        }
        addVisualFeedback("#ff6b9d");
        console.log('語音識別已啟動');
    };
    
    recognition.onend = () => {
        console.log('語音識別結束，當前狀態:', { isVoiceActive, retryCount });
        
        // 只有在用戶未主動關閉且重試次數未超限時才重啟
        if (isVoiceActive && retryCount < MAX_RETRIES) {
            retryCount++;
            console.log(`嘗試重啟語音識別 (${retryCount}/${MAX_RETRIES})`);
            
            // 延遲重啟以避免衝突
            setTimeout(() => {
                try {
                    if (isVoiceActive) {
                        recognition.start();
                    }
                } catch (e) {
                    console.log('語音識別重啟失敗:', e);
                    if (e.name === 'InvalidStateError') {
                        // 忽略 "already started" 錯誤
                        return;
                    }
                    
                    if (retryCount >= MAX_RETRIES) {
                        console.log('達到最大重試次數，停止語音識別');
                        isVoiceActive = false;
                        updateVoiceButton();
                        if (voiceStatus) voiceStatus.style.display = 'none';
                        
                        // 通知用戶
                        confDetails.innerHTML = `
                            <span style="color: #f39c12">
                                🎙️ 語音識別暫時關閉<br>
                                <small>麥克風權限可能已被其他應用佔用</small>
                            </span>
                        `;
                        setTimeout(() => {
                            if (!isVoiceActive) {
                                confDetails.innerText = "請在畫布上書寫數字";
                            }
                        }, 3000);
                    }
                }
            }, 1000); // 1秒後重試
        } else {
            // 用戶主動關閉或達到最大重試次數
            updateVoiceButton();
            if (voiceStatus) voiceStatus.style.display = 'none';
        }
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        console.log("語音識別結果:", transcript);
        
        // 重置重試計數
        retryCount = 0;
        
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
            confDetails.innerHTML = `<b>語音輸入：</b><span style="color:#ff6b9d">${transcript}</span>`;
            addVisualFeedback("#ff6b9d");
        } else {
            // 顯示其他語音指令
            confDetails.innerHTML = `<b>語音指令：</b><span style="color:#ff6b9d">${transcript}</span>`;
        }
    };
    
    recognition.onerror = (event) => {
        console.log("語音識別錯誤:", event.error);
        
        // 根據錯誤類型處理
        switch (event.error) {
            case 'not-allowed':
            case 'audio-capture':
                alert("請允許瀏覽器使用麥克風權限");
                isVoiceActive = false;
                updateVoiceButton();
                if (voiceStatus) voiceStatus.style.display = 'none';
                break;
                
            case 'network':
                console.log('網路錯誤，將嘗試重連');
                break;
                
            case 'no-speech':
                // 無語音輸入，繼續監聽
                break;
                
            default:
                console.log('其他語音錯誤:', event.error);
        }
    };
}

function updateVoiceButton() {
    const voiceBtn = document.getElementById('voiceBtn');
    if (!voiceBtn) return;
    
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
        // 用戶主動關閉
        isVoiceActive = false;
        try {
            recognition.stop();
        } catch (e) {
            console.log('停止語音識別時出錯:', e);
        }
        updateVoiceButton();
        if (voiceStatus) {
            voiceStatus.style.display = 'none';
            voiceStatus.innerHTML = '<span class="pulse-icon">🎙️</span> 正在聆聽語音指令...';
        }
        addVisualFeedback("#34495e");
        console.log('用戶手動關閉語音識別');
    } else {
        // 用戶嘗試開啟
        try {
            // 先檢查麥克風權限
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    // 停止測試流
                    stream.getTracks().forEach(track => track.stop());
                    
                    // 啟動語音識別
                    isVoiceActive = true;
                    recognition.start();
                    updateVoiceButton();
                    addVisualFeedback("#ff6b9d");
                    console.log('用戶手動開啟語音識別');
                })
                .catch(err => {
                    console.log("麥克風權限錯誤:", err);
                    alert("請允許使用麥克風以啟用語音輸入功能");
                    isVoiceActive = false;
                    updateVoiceButton();
                });
        } catch (e) {
            console.log("語音識別啟動錯誤:", e);
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

// ==================== 事件監聽器綁定 ====================

function setupEventListeners() {
    // 畫布事件
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // 觸控事件
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        if (e.touches.length === 1) startDrawing(e);
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (e.touches.length === 1) draw(e);
    });
    canvas.addEventListener('touchend', stopDrawing);
    
    // 按鈕事件
    const buttons = {
        '.btn-run': () => predict(false),
        '.btn-clear': clearCanvas,
        '#eraserBtn': toggleEraser,
        '#camToggleBtn': toggleCamera,
        '#voiceBtn': toggleVoice,
        '.btn-upload': triggerFile
    };
    
    Object.entries(buttons).forEach(([selector, handler]) => {
        const element = document.querySelector(selector);
        if (element) {
            element.addEventListener('click', handler);
        }
    });
    
    // 檔案上傳事件
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFile);
    }
}

// ==================== 頁面載入時初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM 載入完成，開始初始化...');
    setupEventListeners();
    init();
});

// ==================== 錯誤處理和調試 ====================
window.addEventListener('error', function(e) {
    console.error('全局錯誤:', e.error);
    if (confDetails) {
        confDetails.innerHTML = `<span style="color: #ff4d4d">系統錯誤: ${e.message}</span>`;
    }
});

// TensorFlow.js 內存監控
setInterval(() => {
    try {
        const memoryInfo = tf.memory();
        if (memoryInfo.numTensors > 100) {
            console.warn(`TensorFlow.js 內存警告: ${memoryInfo.numTensors} 個張量`);
        }
    } catch (e) {
        // 忽略內存檢查錯誤
    }
}, 10000);
