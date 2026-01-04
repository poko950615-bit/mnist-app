/**
 * 🌌 銀河手寫數字辨識系統 - 純前端終極版
 * 完全獨立運行，無需後端伺服器
 * 使用 TensorFlow.js 在瀏覽器中執行 AI 辨識
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

// ==================== 模型加載 ====================
async function loadModel() {
    try {
        confDetails.innerText = "🌌 正在啟動銀河辨識引擎...";
        
        // 等待 TensorFlow.js 準備就緒
        await tf.ready();
        console.log('TensorFlow.js 版本:', tf.version.tfjs);
        
        // 設置後端（優先使用 CPU 以確保穩定）
        await tf.setBackend('cpu');
        console.log('使用後端:', tf.getBackend());
        
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
        
        confDetails.innerText = "🚀 系統就緒，請開始書寫數字";
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

// ==================== 影像處理核心（移植自 p.py）====================
// ... [這裡插入你完整的影像處理函數，包括：]
// imageDataToGrayArray, calculateAverageBrightness, invertBackground,
// simpleGaussianBlur, calculateOtsuThreshold, binarizeImage,
// findConnectedComponents, advancedPreprocess 等所有函數
// 確保這是完整的移植，不要省略任何函數

// ==================== 主辨識函數 ====================
async function predict(isRealtime = false) {
    // 防止重複處理
    if (isProcessing) {
        console.log('⏳ 正在處理中，跳過本次請求');
        return;
    }
    
    isProcessing = true;
    
    try {
        // 檢查模型
        if (!model) {
            console.log('模型未載入，嘗試載入...');
            const loaded = await loadModel();
            if (!loaded) {
                digitDisplay.innerText = "❌";
                confDetails.innerHTML = "<b>錯誤：</b>模型未載入";
                isProcessing = false;
                return;
            }
        }
        
        // 顯示載入狀態
        if (!isRealtime) {
            digitDisplay.innerHTML = '<span class="pulse-icon">🌠</span>';
            confDetails.innerText = "正在分析影像...";
        }
        
        // ... [這裡插入完整的 predict 函數邏輯，包括：]
        // 1. 獲取畫布影像
        // 2. 影像預處理（灰階、反轉、模糊、二值化）
        // 3. 連通域分析與過濾
        // 4. 數字識別（包含連體字分割）
        // 5. 顯示結果
        // 6. 即時模式的框繪製
        
    } catch (error) {
        console.error('辨識錯誤:', error);
        digitDisplay.innerText = "❌";
        confDetails.innerHTML = `<b>錯誤：</b>${error.message}`;
        addVisualFeedback("#e74c3c");
    } finally {
        isProcessing = false;
    }
}

// ==================== UI 互動功能 ====================
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
    const eraserBtn = document.getElementById('eraserBtn');
    if (eraserBtn) {
        eraserBtn.innerText = isEraser ? "🧽 橡皮擦：開啟" : "🧽 橡皮擦：關閉";
        eraserBtn.classList.toggle('eraser-active', isEraser);
    }
    updatePen();
    addVisualFeedback(isEraser ? "#e74c3c" : "#f39c12");
}

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

// ==================== 相機功能 ====================
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
        addVisualFeedback("#9b59b6");
        
    } catch (err) {
        console.error('鏡頭啟動失敗:', err);
        alert("無法啟動鏡頭：請確保已授予相機權限");
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
    
    video.style.display = "none";
    document.getElementById('mainBox').classList.remove('cam-active');
    
    const camToggleBtn = document.getElementById('camToggleBtn');
    if (camToggleBtn) {
        camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 開啟鏡頭';
    }
    
    init(); // 重新初始化畫布
    addVisualFeedback("#34495e");
}

// ==================== 檔案上傳 ====================
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

// ==================== 語音功能 ====================
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
    
    recognition.onstart = () => {
        isVoiceActive = true;
        updateVoiceButton();
        if (voiceStatus) voiceStatus.style.display = 'block';
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
                if (voiceStatus) voiceStatus.style.display = 'none';
            }
        } else {
            updateVoiceButton();
            if (voiceStatus) voiceStatus.style.display = 'none';
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
            confDetails.innerHTML = `<b>語音輸入：</b><span style="color:#ff6b9d">${transcript}</span>`;
            addVisualFeedback("#ff6b9d");
        }
    };
    
    recognition.onerror = (event) => {
        console.log("語音識別錯誤:", event.error);
        if (event.error === 'not-allowed' || event.error === 'audio-capture') {
            alert("請允許瀏覽器使用麥克風權限");
            isVoiceActive = false;
            updateVoiceButton();
            if (voiceStatus) voiceStatus.style.display = 'none';
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
        isVoiceActive = false;
        recognition.stop();
        updateVoiceButton();
        if (voiceStatus) voiceStatus.style.display = 'none';
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
}

function draw(e) {
    e.preventDefault();
    if (!isDrawing) return;
    
    const { x, y } = getCanvasCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        if (!cameraStream) {
            setTimeout(() => predict(false), 300);
        }
    }
}

// ==================== 事件監聽器 ====================
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
