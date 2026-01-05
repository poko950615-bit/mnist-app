/**
 * 🌌 銀河手寫數字辨識系統 - 終極相機強化版
 * * 修改重點：
 * 1. [相機核心] 導入 ROI (Region of Interest) 掃描框技術，徹底排除環境背景干擾。
 * 2. [相機核心] 提升辨識頻率至 100ms (極速響應)。
 * 3. [相機核心] 加入「結果穩定器」，防止數字跳動。
 * 4. [修復] 保留了之前的繪圖斷線修復與上傳 Bug 修復。
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

// 繪圖座標記錄
let lastX = 0;
let lastY = 0;

// 相機模式專用變數
let lastPredicationTime = 0;
const PREDICTION_INTERVAL = 100; // 100ms 極速辨識
const STABILITY_THRESHOLD = 2;   // 連續偵測到 2 次才顯示（防閃爍）
let predictionHistory = [];      // 辨識結果歷史紀錄

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
            
            const traverseAndPatch = (obj) => {
                if (!obj || typeof obj !== 'object') return;
                if (obj.class_name === 'InputLayer' && obj.config) {
                    const cfg = obj.config;
                    if (!cfg.batchInputShape && !cfg.batch_input_shape) {
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

            if (artifacts.weightSpecs) {
                artifacts.weightSpecs.forEach(spec => {
                    if (spec.name.includes('sequential/')) {
                        spec.name = spec.name.replace('sequential/', '');
                    }
                });
            }
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
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    
    initSpeechRecognition();
    await loadModel();
    
    digitDisplay.innerText = "---";
    confDetails.innerText = "🚀 系統就緒，請開始書寫數字";
    
    addGalaxyEffects();
    console.log('✅ 系統初始化完成');
}

// ==================== 模型加載 ====================
async function loadModel() {
    try {
        confDetails.innerText = "🌌 正在啟動銀河辨識引擎...";
        const availableBackends = tf.engine().backendNames;
        let backendToUse = 'cpu';
        try {
            const tempCanvas = document.createElement('canvas');
            const gl = tempCanvas.getContext('webgl2') || tempCanvas.getContext('webgl');
            if (gl) backendToUse = 'webgl';
        } catch (e) { console.log('WebGL 不可用'); }
        
        await tf.setBackend(backendToUse);
        await tf.ready();
        
        const modelUrl = 'tfjs_model/model.json';
        model = await tf.loadLayersModel(new PatchModelLoader(modelUrl));
        
        // 模型暖身
        const testInput = tf.zeros([1, 28, 28, 1]);
        const testOutput = model.predict(testInput);
        await testOutput.data();
        testInput.dispose();
        testOutput.dispose();
        
        confDetails.innerText = tf.getBackend() === 'webgl' ? "🚀 系統就緒（WebGL加速）" : "🚀 系統就緒（CPU模式）";
        return true;
    } catch (error) {
        console.error('❌ 模型載入失敗:', error);
        confDetails.innerHTML = `<span style="color: #ff4d4d">❌ 模型載入失敗: ${error.message}</span>`;
        return false;
    }
}

// ==================== 影像處理核心 (保持原算法) ====================
function imageDataToGrayArray(imageData) {
    const { width, height, data } = imageData;
    const grayArray = new Uint8Array(width * height);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        grayArray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    return { data: grayArray, width, height };
}

function calculateAverageBrightness(grayArray) {
    let sum = 0;
    for (let i = 0; i < grayArray.data.length; i++) sum += grayArray.data[i];
    return sum / grayArray.data.length;
}

function invertBackground(grayArray) {
    const inverted = new Uint8Array(grayArray.data.length);
    for (let i = 0; i < grayArray.data.length; i++) inverted[i] = 255 - grayArray.data[i];
    return { data: inverted, width: grayArray.width, height: grayArray.height };
}

function simpleGaussianBlur(grayArray) {
    const { data, width, height } = grayArray;
    const result = new Uint8Array(width * height);
    const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    const kernelSum = 16;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sum = 0, k = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    sum += data[(y + ky) * width + (x + kx)] * kernel[k++];
                }
            }
            result[y * width + x] = Math.round(sum / kernelSum);
        }
    }
    return { data: result, width, height };
}

function calculateOtsuThreshold(grayArray) {
    const { data } = grayArray;
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i++) histogram[data[i]]++;
    const total = data.length;
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * histogram[i];
    let sumB = 0, wB = 0, wF = 0, maxVariance = 0, threshold = 0;
    for (let i = 0; i < 256; i++) {
        wB += histogram[i];
        if (wB === 0) continue;
        wF = total - wB;
        if (wF === 0) break;
        sumB += i * histogram[i];
        const mB = sumB / wB;
        const mF = (sum - sumB) / wF;
        const variance = wB * wF * Math.pow(mB - mF, 2);
        if (variance > maxVariance) { maxVariance = variance; threshold = i; }
    }
    return threshold;
}

function binarizeImage(grayArray, threshold) {
    const binary = new Uint8Array(grayArray.data.length);
    for (let i = 0; i < grayArray.data.length; i++) binary[i] = grayArray.data[i] > threshold ? 255 : 0;
    return { data: binary, width: grayArray.width, height: grayArray.height };
}

function findConnectedComponents(binaryImage) {
    const { data, width, height } = binaryImage;
    const visited = new Array(width * height).fill(false);
    const components = [];
    const directions = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]];
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (!visited[idx] && data[idx] === 255) {
                const queue = [[x, y]];
                visited[idx] = true;
                let minX = x, maxX = x, minY = y, maxY = y, area = 0;
                const pixels = [];
                while (queue.length > 0) {
                    const [cx, cy] = queue.shift();
                    area++;
                    pixels.push([cx, cy]);
                    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
                    minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);
                    for (const [dx, dy] of directions) {
                        const nx = cx + dx, ny = cy + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            if (!visited[nIdx] && data[nIdx] === 255) {
                                visited[nIdx] = true;
                                queue.push([nx, ny]);
                            }
                        }
                    }
                }
                const w = maxX - minX + 1, h = maxY - minY + 1;
                components.push({ x: minX, y: minY, w: w, h: h, area: area, aspectRatio: w / h, solidity: area / (w * h), pixels: pixels });
            }
        }
    }
    return components;
}

function calculateImageMoments(binaryImage) {
    const { data, width, height } = binaryImage;
    let m00 = 0, m10 = 0, m01 = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (data[idx] > 0) {
                const val = data[idx] / 255;
                m00 += val; m10 += x * val; m01 += y * val;
            }
        }
    }
    return { m00, m10, m01 };
}

function advancedPreprocess(roiImage) {
    const { data, width, height } = roiImage;
    const binaryArray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i++) binaryArray[i] = data[i] > 128 ? 255 : 0;
    
    // 膨脹處理，增強筆畫連通性
    const dilated = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let maxVal = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const nx = x + kx, ny = y + ky;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) maxVal = Math.max(maxVal, binaryArray[ny * width + nx]);
                }
            }
            dilated[y * width + x] = maxVal;
        }
    }
    
    // Padding
    const pad = Math.floor(Math.max(height, width) * 0.45);
    const pw = width + 2 * pad, ph = height + 2 * pad;
    const paddedData = new Uint8Array(pw * ph);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) paddedData[(y + pad) * pw + (x + pad)] = dilated[y * width + x];
    }
    
    // 縮放到 28x28
    const targetSize = 28;
    const scaledData = new Uint8Array(targetSize * targetSize);
    for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) scaledData[y * targetSize + x] = paddedData[Math.floor(y * (ph / targetSize)) * pw + Math.floor(x * (pw / targetSize))];
    }
    
    // 質心校正 (Centering)
    const moments = calculateImageMoments({ data: scaledData, width: targetSize, height: targetSize });
    const finalData = new Float32Array(targetSize * targetSize);
    if (moments.m00 !== 0) {
        const dx = 14 - (moments.m10 / moments.m00), dy = 14 - (moments.m01 / moments.m00);
        for (let y = 0; y < targetSize; y++) {
            for (let x = 0; x < targetSize; x++) {
                const sx = Math.round(x - dx), sy = Math.round(y - dy);
                if (sx >= 0 && sx < targetSize && sy >= 0 && sy < targetSize) finalData[y * targetSize + x] = scaledData[sy * targetSize + sx] / 255.0;
            }
        }
    } else {
        for (let i = 0; i < scaledData.length; i++) finalData[i] = scaledData[i] / 255.0;
    }
    return finalData;
}

// ==================== [核心修改] 辨識與預測函數 ====================

// 輔助：繪製 ROI 掃描框
function drawROIGuide(ctx, width, height, roi) {
    // 1. 整個畫面變暗 (半透明黑)
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(0, 0, width, height);

    // 2. 挖出中間的洞 (清除半透明層)
    ctx.clearRect(roi.x, roi.y, roi.w, roi.h);

    // 3. 畫綠色掃描框
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 4;
    ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);

    // 4. 文字提示
    ctx.fillStyle = "#00FF00";
    ctx.font = "bold 20px Arial";
    ctx.fillText("請將數字置於框內", roi.x + 20, roi.y - 15);
}

// 主辨識函數
async function predict(isRealtime = false) {
    if (isProcessing || !model) return;
    
    // 頻率限制 (僅針對 Realtime 模式)
    const now = Date.now();
    if (isRealtime && (now - lastPredicationTime < PREDICTION_INTERVAL)) return;
    lastPredicationTime = now;

    isProcessing = true;
    try {
        // --- 準備畫布 ---
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width; tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext('2d');

        // 定義掃描框 (ROI) - 畫布中心 300x300 的區域
        const roiSize = 300;
        const roi = {
            x: (canvas.width - roiSize) / 2,
            y: (canvas.height - roiSize) / 2,
            w: roiSize,
            h: roiSize
        };

        if (cameraStream && isRealtime) {
            // 繪製相機影像
            tempCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
            // 繪製綠色掃描框 UI 供使用者參考 (注意：這是畫在記憶體中的 canvas，不會影響辨識，但我們需要同步更新到主畫布給使用者看)
            const mainCtx = canvas.getContext('2d');
            mainCtx.clearRect(0, 0, canvas.width, canvas.height);
            mainCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
            drawROIGuide(mainCtx, canvas.width, canvas.height, roi);
        } else {
            // 一般手寫模式，讀取整個畫布
            tempCtx.drawImage(canvas, 0, 0);
        }

        // --- 擷取影像資料 ---
        // 關鍵修改：如果是即時模式，只擷取 ROI 區域的像素！完全排除外部環境
        let imageData;
        if (isRealtime) {
            imageData = tempCtx.getImageData(roi.x, roi.y, roi.w, roi.h);
        } else {
            imageData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
        }

        // --- 影像預處理 pipeline ---
        const grayImage = imageDataToGrayArray(imageData);
        const avgBrightness = calculateAverageBrightness(grayImage);
        
        // 自動判斷是否反轉 (紙張通常是白底黑字，模型需要黑底白字)
        const processedGray = avgBrightness > 100 ? invertBackground(grayImage) : grayImage;
        const blurred = simpleGaussianBlur(processedGray);
        const threshold = calculateOtsuThreshold(blurred);
        const binaryImage = binarizeImage(blurred, threshold);

        // --- [相機模式專用] 雜訊過濾 ---
        if (isRealtime) {
            // 計算白色像素比例
            let whiteCount = 0;
            for(let i=0; i<binaryImage.data.length; i++) if(binaryImage.data[i] === 255) whiteCount++;
            const whiteRatio = whiteCount / binaryImage.data.length;

            // 如果畫面太乾淨(全黑)或太雜亂(全白)，直接放棄
            if (whiteRatio < 0.01 || whiteRatio > 0.4) {
                digitDisplay.innerText = "---";
                predictionHistory = []; // 重置歷史
                isProcessing = false;
                return;
            }
        }

        // --- 連通域分析 ---
        const components = findConnectedComponents(binaryImage);
        
        // 過濾邏輯
        const MIN_AREA = isRealtime ? 300 : 150; // 相機模式需要更大的有效面積
        const filtered = components.filter(c => {
            // 1. 面積檢查
            if (c.area < MIN_AREA) return false;
            // 2. 形狀檢查 (數字不會太扁長)
            if (c.aspectRatio > 3.0 || c.aspectRatio < 0.15) return false;
            // 3. 實心度檢查
            if (c.solidity < 0.12) return false;

            // 4. [相機模式] 邊緣接觸檢查
            // 如果物件碰到 ROI 的邊框，代表數字沒拍完整，忽略
            if (isRealtime) {
                const border = 5;
                if (c.x < border || c.y < border || 
                   (c.x + c.w) > (imageData.width - border) || 
                   (c.y + c.h) > (imageData.height - border)) {
                    return false;
                }
            }
            return true;
        });

        // 排序：相機模式只取最大的那個(假設使用者會把數字放中間)，手寫模式取左到右
        if (isRealtime) {
            filtered.sort((a, b) => b.area - a.area);
            // 只留最大的一個
            if (filtered.length > 1) filtered.length = 1;
        } else {
            filtered.sort((a, b) => a.x - b.x);
        }

        let finalResult = "";
        const details = [];

        // --- 開始辨識 ---
        for (const comp of filtered) {
            const roiData = { data: new Uint8Array(comp.w * comp.h), width: comp.w, height: comp.h };
            for (let y = 0; y < comp.h; y++) {
                for (let x = 0; x < comp.w; x++) {
                    // 注意：這裡的 binaryImage 座標已經是相對 ROI 的
                    roiData.data[y * comp.w + x] = binaryImage.data[(comp.y + y) * binaryImage.width + (comp.x + x)];
                }
            }

            const processedData = advancedPreprocess(roiData);
            const tensor = tf.tensor4d(processedData, [1, 28, 28, 1]);
            const prediction = model.predict(tensor);
            const scores = await prediction.data();
            const digit = prediction.argMax(-1).dataSync()[0];
            const confidence = Math.max(...scores);
            tensor.dispose(); prediction.dispose();

            // [相機模式] 極高信心度門檻，排除雜訊
            const CONF_THRESHOLD = isRealtime ? 0.95 : 0.7;

            if (confidence > CONF_THRESHOLD) {
                finalResult += digit.toString();
                details.push({ digit, conf: `${(confidence * 100).toFixed(1)}%` });
            }
        }

        // --- 結果處理與穩定顯示 ---
        if (finalResult) {
            if (isRealtime) {
                // 穩定器邏輯：連續 N 次看到一樣的數字才顯示
                predictionHistory.push(finalResult);
                if (predictionHistory.length > STABILITY_THRESHOLD) predictionHistory.shift();
                
                // 檢查歷史紀錄是否都一樣
                const allSame = predictionHistory.every(v => v === finalResult);
                
                if (allSame && predictionHistory.length === STABILITY_THRESHOLD) {
                    digitDisplay.innerText = finalResult;
                    addVisualFeedback("#2ecc71");
                    confDetails.innerText = `相機鎖定: ${details[0].digit} (${details[0].conf})`;
                    
                    // 在相機畫面上標示出偵測到的框 (相對於 ROI)
                    const mainCtx = canvas.getContext('2d');
                    const comp = filtered[0];
                    if (comp) {
                        mainCtx.strokeStyle = "#FFFF00";
                        mainCtx.lineWidth = 3;
                        // 還原回主畫布座標：ROI起始 + 組件偏移
                        mainCtx.strokeRect(roi.x + comp.x, roi.y + comp.y, comp.w, comp.h);
                    }
                }
            } else {
                // 手寫模式直接顯示
                digitDisplay.innerText = finalResult;
                updateDetails(details);
                addVisualFeedback("#2ecc71");
            }
        } else {
            // 沒辨識到
            if (isRealtime) {
                 predictionHistory = []; // 斷掉連續紀錄
                 digitDisplay.innerText = "---";
                 confDetails.innerText = "正在掃描...";
            } else {
                digitDisplay.innerText = "---";
                confDetails.innerText = "未偵測到有效數字";
            }
        }

        isProcessing = false;
        return { full_digit: finalResult };

    } catch (error) {
        console.error("辨識錯誤:", error);
        isProcessing = false;
        return { error: error.message };
    }
}

// ==================== UI 與工具功能 ====================

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
    const btn = document.getElementById('eraserBtn');
    if (btn) {
        btn.innerText = isEraser ? "🧽 橡皮擦：開啟" : "🧽 橡皮擦：關閉";
        btn.classList.toggle('eraser-active', isEraser);
    }
    updatePen();
    if (isEraser) addVisualFeedback("#e74c3c");
}

function clearCanvas() {
    // 只有在非相機模式下才清除顯示
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!cameraStream) {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        digitDisplay.innerText = "---";
        confDetails.innerText = "🪐 畫布已清空，請重新書寫";
    }
    addVisualFeedback("#2ecc71");
    addGalaxyEffects();
}

// [修正] 相機開關邏輯
async function toggleCamera() {
    if (cameraStream) {
        stopCamera();
    } else {
        try {
            // 請求高清串流以利辨識
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false
            });
            video.srcObject = cameraStream;
            video.play(); // 確保影片播放
            video.style.display = "block"; // 隱藏原生 video 元素，我們畫在 canvas 上
            video.style.opacity = "0";     // 但保持它運作

            document.getElementById('mainBox').classList.add('cam-active');
            
            const btn = document.getElementById('camToggleBtn');
            if(btn) btn.innerHTML = '<span class="btn-icon">📷</span> 關閉鏡頭';
            
            // 使用更頻繁的 Loop 進行即時辨識 (100ms 一次)
            realtimeInterval = setInterval(() => predict(true), PREDICTION_INTERVAL);
            
            addVisualFeedback("#9b59b6");
            confDetails.innerText = "📷 相機已啟動，請將數字對準綠框";
        } catch (err) {
            alert("無法啟動鏡頭：請確保已授予相機權限");
            console.error(err);
        }
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    if (realtimeInterval) { 
        clearInterval(realtimeInterval); 
        realtimeInterval = null; 
    }
    
    // 恢復 UI 狀態
    video.style.display = "none";
    document.getElementById('mainBox').classList.remove('cam-active');
    
    const btn = document.getElementById('camToggleBtn');
    if(btn) btn.innerHTML = '<span class="btn-icon">📷</span> 開啟鏡頭';
    
    init(); // 恢復黑底畫布供手寫
}

// [修正] 檔案上傳
function triggerFile() {
    document.getElementById('fileInput').click();
}

function handleFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (cameraStream) stopCamera();
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            clearCanvas();
            const ratio = Math.min(canvas.width / img.width * 0.8, canvas.height / img.height * 0.8);
            const w = img.width * ratio, h = img.height * ratio;
            ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
            predict(false);
            
            // [關鍵修正] 清空 input value，確保可重複上傳
            event.target.value = ""; 
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ==================== 語音功能 ====================
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-TW';
    recognition.continuous = true;
    recognition.interimResults = false;
    
    recognition.onstart = () => {
        isVoiceActive = true;
        updateVoiceButton();
        if (voiceStatus) {
            voiceStatus.style.display = 'block';
            voiceStatus.innerHTML = '<span class="pulse-icon">🎙️</span> 聆聽中...';
        }
    };
    
    recognition.onend = () => {
        if (isVoiceActive) {
            setTimeout(() => { 
                if (isVoiceActive && recognition) {
                    try { recognition.start(); } catch(e) {}
                } 
            }, 1000);
        } else {
            updateVoiceButton();
            if (voiceStatus) voiceStatus.style.display = 'none';
        }
    };

    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        if (transcript.includes('清除') || transcript.includes('清空')) clearCanvas();
        else if (transcript.includes('辨識') || transcript.includes('開始')) predict(false);
        else if (transcript.includes('鏡頭')) toggleCamera();
        else if (transcript.includes('橡皮擦')) toggleEraser();
    };
}

function toggleVoice() {
    if (!recognition) { alert("瀏覽器不支援語音"); return; }
    if (isVoiceActive) {
        isVoiceActive = false;
        recognition.stop();
    } else {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(() => {
            isVoiceActive = true;
            recognition.start();
        }).catch(() => alert("請開啟麥克風權限"));
    }
}

function updateVoiceButton() {
    const btn = document.getElementById('voiceBtn');
    if (!btn) return;
    btn.innerHTML = isVoiceActive ? '<span class="btn-icon">🌌</span> 語音輸入：開啟' : '<span class="btn-icon">🌌</span> 語音輸入：關閉';
    btn.classList.toggle('voice-active', isVoiceActive);
}

// ==================== [修正] 繪圖事件 (解決起點連線問題) ====================
function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
}

function startDrawing(e) {
    // 若在相機模式，禁止繪圖以免干擾
    if (cameraStream) return;

    e.preventDefault();
    const { x, y } = getCanvasCoordinates(e);
    isDrawing = true;
    
    // [關鍵] 斷開與原點的連結
    ctx.beginPath();
    ctx.moveTo(x, y);
    
    lastX = x;
    lastY = y;
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const { x, y } = getCanvasCoordinates(e);
    
    ctx.lineTo(x, y);
    ctx.stroke();
    
    // 保持連續性
    ctx.beginPath();
    ctx.moveTo(x, y);
    
    lastX = x;
    lastY = y;
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.closePath();
        // 手寫模式下，畫完稍微延遲後自動辨識
        if (!cameraStream) setTimeout(() => predict(false), 300);
    }
}

// ==================== 事件綁定與初始化 ====================
function setupEventListeners() {
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);
    
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);

    document.querySelector('.btn-run')?.addEventListener('click', () => predict(false));
    document.querySelector('.btn-clear')?.addEventListener('click', clearCanvas);
    document.getElementById('eraserBtn')?.addEventListener('click', toggleEraser);
    document.getElementById('camToggleBtn')?.addEventListener('click', toggleCamera);
    document.getElementById('voiceBtn')?.addEventListener('click', toggleVoice);
    document.querySelector('.btn-upload')?.addEventListener('click', triggerFile);
    document.getElementById('fileInput')?.addEventListener('change', handleFile);
}

function addVisualFeedback(color) {
    const btns = document.querySelectorAll('button');
    btns.forEach(b => {
        const originalShadow = b.style.boxShadow;
        b.style.boxShadow = `0 0 15px ${color}`;
        setTimeout(() => b.style.boxShadow = originalShadow, 300);
    });
}

function addGalaxyEffects() {
    ctx.fillStyle = "rgba(163, 217, 255, 0.3)";
    ctx.beginPath(); ctx.arc(650, 20, 2, 0, Math.PI*2); ctx.fill();
    updatePen();
}

function updateDetails(data) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (!data.length) html += "未偵測到有效數字";
    else data.forEach((item, i) => {
        html += `數字 ${i + 1}: <b style="color:#a3d9ff">${item.digit}</b> (信心度: ${item.conf})<br>`;
    });
    confDetails.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    init();
});
