let currentSession = null;
let isDebugMode = false; 
let lastGeneratedWordData = null; 
let isColorInverted = true; 
let isCurveEnabled = true; 

const chars = 'אבגדהוזחטיכךלמםנןסעפףצץקרשת.,-!?/)( ';
const charToLabel = Object.fromEntries([...chars].map((c, i) => [c, i]));

const processor = new ImageProcessor(0.9, 2); 

const originalCurve = processor.applyContrastCurve.bind(processor);
processor.applyContrastCurve = function(val) {
    if (isCurveEnabled) return originalCurve(val);
    let x = Math.max(0, Math.min(1, (val + 1.0) / 2.0));
    let pixelValue = Math.floor((1.0 - x) * 255);
    return Math.max(0, Math.min(255, pixelValue));
};

const LINE_HEIGHT_PX = 28; 

function getLetterBounds(char) {
    let top = 1, bottom = 0, sd = 0.05; 
    
    if ('לטצץף'.includes(char)) { top = 1.6; bottom = 0; sd = 0.1; }
    else if ('ךןק'.includes(char)) { top = 1; bottom = -0.6; sd = 0.1; }
    else if ('ת'.includes(char)) { top = 1; bottom = -0.6; sd = 0.1; } 
    else if ('י'.includes(char)) { top = 1; bottom = 0.6; sd = 0.05; }
    else if ('\'"'.includes(char)) { top = 1.2; bottom = 0.8; sd = 0.05; }
    else if ('.,:;'.includes(char)) { top = 0.35; bottom = 0; sd = 0.02; }
    else if ('-'.includes(char)) { top = 0.6; bottom = 0.4; sd = 0.02; }
    else if ('/'.includes(char)) { top = 1.2; bottom = -0.2; sd = 0.05; }
    else if ('()!?'.includes(char)) { top = 1.2; bottom = -0.2; sd = 0.1; }
    
    return { top, bottom, sd };
}

// --- NEW: KERNING ENGINE ---
function getKerningRatio(char) {
    // Defines how much the letter's invisible slot shrinks, allowing surrounding letters to tuck in.
    if ('ל'.includes(char)) return 0.40; // 40% overlap! Lets letters hide right under the sweeping arm.
    if ('ףץע'.includes(char)) return 0.25; // 25% overlap for other wide-sweeping letters.
    if ('ויזןך'.includes(char)) return 0.05; // Very narrow letters shouldn't be squished too much.
    if ('.,:;\'"'.includes(char)) return 0.0; // Punctuation keeps its full tiny width.
    return 0.15; // 15% overlap for everything else creates a connected cursive feel.
}

function showStatus(text, iconName) {
    const container = document.getElementById('statusContainer');
    document.getElementById('statusIcon').src = `assets/${iconName}`;
    document.getElementById('statusText').innerText = text;
    container.classList.remove('hidden');
}

function hideStatus() { document.getElementById('statusContainer').classList.add('hidden'); }

async function init() {
    try {
        currentSession = await ort.InferenceSession.create(`./hebrew_gan_bundled_v7.2.4.onnx?v=${Date.now()}`);
        generateWord(); 
    } catch (e) { showStatus("שגיאה בטעינת המודל", "error.png"); }
}

function refreshSeed() { generateWord(); }
function toggleDebug() { isDebugMode = !isDebugMode; if (lastGeneratedWordData) drawCanvas(); }

function toggleCurve() {
    isCurveEnabled = !isCurveEnabled;
    if (lastGeneratedWordData) {
        for (let i = 0; i < lastGeneratedWordData.items.length; i++) {
            let item = lastGeneratedWordData.items[i];
            if (item.type === 'char' && item.rawPixels) {
                let newCropInfo = processor.processAndCrop(item.rawPixels);
                Object.assign(item, newCropInfo);
            }
        }
        drawCanvas();
    }
}

function toggleInvert() { isColorInverted = !isColorInverted; if (lastGeneratedWordData) drawCanvas(); }

function getGaussian(mean, sd) {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); while(v === 0) v = Math.random();
    return Math.round((Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * sd) + mean);
}

function getGaussianFloat(mean, sd) {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); while(v === 0) v = Math.random();
    return (Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * sd) + mean;
}

function generateNoiseVector() {
    let noise = new Float32Array(100);
    for (let i = 0; i < 100; i++) {
        let u = 0, v = 0;
        while(u === 0) u = Math.random(); while(v === 0) v = Math.random();
        noise[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
    return noise;
}

// === PHASE 1: GENERATION ===
async function generateWord() {
    if (!currentSession) return;
    hideStatus();

    const rawText = document.getElementById('textInput').value;
    const text = rawText.replace(/[^א-ת.,\-!?/)( :;'"']/g, ''); 
    if (!text) return showStatus("אנא הזן תווים חוקיים", "exclamation.png");

    const canvas = document.getElementById('outputCanvas');
    canvas.classList.remove('loaded');
    
    setTimeout(async () => {
        const generatedItems = [];

        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            if (char === ' ') {
                generatedItems.push({ type: 'space', width: Math.max(5, getGaussian(25, 7)), char: ' ' });
                continue;
            }

            let modelChar = char;
            if (char === ':') modelChar = '.';
            if (char === "'" || char === '"' || char === ';') modelChar = ',';

            const labelInt = charToLabel[modelChar] !== undefined ? charToLabel[modelChar] : 0; 
            const labelTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(labelInt)]), [1]);

            let validLetter = false; let attempts = 0; let cropInfo = null;
            let rawPixels = null;

            while (!validLetter && attempts < 10) {
                try {
                    let noiseTensor = new ort.Tensor('float32', generateNoiseVector(), [1, 100]);
                    const results = await currentSession.run({ noise: noiseTensor, label: labelTensor });
                    
                    rawPixels = new Float32Array(results.generated_image.data); 
                    cropInfo = processor.processAndCrop(rawPixels);
                    
                    if (cropInfo.isEmpty) attempts++; else validLetter = true;
                } catch (e) { break; }
            }

            if (cropInfo && !cropInfo.isEmpty) {
                generatedItems.push({ type: 'char', char: char, rawPixels: rawPixels, ...cropInfo });
            }
        }

        lastGeneratedWordData = { items: generatedItems, baselineY: 64 - getGaussian(16, 2) };
        drawCanvas();
    }, 50);
}

async function regenerateSingleChar(index) {
    const item = lastGeneratedWordData.items[index];
    
    if (item.type === 'space') {
        item.width = Math.max(5, getGaussian(25, 7));
        item.slotW = item.width; 
        drawCanvas();
        return; 
    }

    let modelChar = item.char;
    if (item.char === ':') modelChar = '.';
    if (item.char === "'" || item.char === '"' || item.char === ';') modelChar = ',';

    const labelInt = charToLabel[modelChar] !== undefined ? charToLabel[modelChar] : 0; 
    const labelTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(labelInt)]), [1]);

    let validLetter = false; let attempts = 0; let cropInfo = null; let rawPixels = null;

    const btn = document.getElementById(`regen-btn-${index}`);
    if(btn) btn.innerHTML = '<span style="font-size:12px;">⏳</span>';

    while (!validLetter && attempts < 10) {
        try {
            let noiseTensor = new ort.Tensor('float32', generateNoiseVector(), [1, 100]);
            const results = await currentSession.run({ noise: noiseTensor, label: labelTensor });
            rawPixels = new Float32Array(results.generated_image.data);
            cropInfo = processor.processAndCrop(rawPixels);
            if (cropInfo.isEmpty) attempts++; else validLetter = true;
        } catch (e) { break; }
    }

    if (cropInfo && !cropInfo.isEmpty) {
        let oldSlotW = item.slotW; 
        let oldJitter = item.heightJitter;
        lastGeneratedWordData.items[index] = { 
            type: 'char', 
            char: item.char, 
            heightJitter: oldJitter, 
            rawPixels: rawPixels,
            ...cropInfo 
        };
        drawCanvas(); 
    }
}

// === PHASE 2: RENDERING ===
function drawCanvas() {
    if (!lastGeneratedWordData) return;

    const canvas = document.getElementById('outputCanvas');
    const wrapper = document.getElementById('canvasWrapper');
    const overlay = document.getElementById('debugOverlay');
    const ctx = canvas.getContext('2d');
    
    overlay.innerHTML = '';
    
    // --- PASS 1: Calculate Tight Bounds & Apply Kerning ---
    let scaledTotalWidth = 0;
    for (let item of lastGeneratedWordData.items) {
        if (item.type === 'space') {
            item.finalW = item.width;
            if (!item.slotW) item.slotW = item.finalW;
        } else {
            let bounds = getLetterBounds(item.char);
            let targetHeightUnits = Math.abs(bounds.top - bounds.bottom);
            
            if (item.heightJitter === undefined) item.heightJitter = getGaussianFloat(0, bounds.sd);
            targetHeightUnits += item.heightJitter;
            
            let targetHeightPx = targetHeightUnits * LINE_HEIGHT_PX;
            let scaleFactor = targetHeightPx / item.cropH;
            
            item.finalH = targetHeightPx;
            item.finalW = item.cropW * scaleFactor;
            item.scaleFactor = scaleFactor;
            item.yPosTop = bounds.top; 
            
            if (item.char === '"') item.finalW = item.finalW * 2; 
            
            // Apply Kerning: The slot is smaller than the visual width!
            if (!item.slotW) {
                let overlapRatio = getKerningRatio(item.char);
                item.slotW = item.finalW * (1 - overlapRatio);
                item.slotW = Math.max(item.slotW, 5); // Failsafe
            }
        }
        scaledTotalWidth += item.slotW;
    }

    // Add extra padding to the canvas so the negative kerning doesn't clip the first letter!
    canvas.width = Math.max(10, scaledTotalWidth + 40);
    
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    wrapper.style.width = canvas.width + 'px'; 
    wrapper.style.height = '64px'; 
    wrapper.style.transform = 'scale(1.3)'; 
    wrapper.style.transformOrigin = 'right center'; 
    wrapper.style.alignSelf = 'flex-start';         
    wrapper.style.marginTop = '30px'; 
    wrapper.style.marginBottom = '20px'; 
    canvas.style.transform = 'none'; 
    
    ctx.fillStyle = isColorInverted ? "#ffffff" : "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Start 20px inside the canvas to allow the first letter to safely bleed to the right
    let currentX = canvas.width - 20; 
    let baseline = lastGeneratedWordData.baselineY;

    if (isDebugMode) {
        ctx.strokeStyle = '#0000FF'; ctx.beginPath(); ctx.moveTo(0, baseline); ctx.lineTo(canvas.width, baseline); ctx.stroke();
        ctx.strokeStyle = '#00BFFF'; ctx.beginPath(); ctx.moveTo(0, baseline - LINE_HEIGHT_PX); ctx.lineTo(canvas.width, baseline - LINE_HEIGHT_PX); ctx.stroke();
    }

    // --- PASS 2: Drawing ---
    for (let i = 0; i < lastGeneratedWordData.items.length; i++) {
        let item = lastGeneratedWordData.items[i];
        
        currentX -= item.slotW; // Move cursor by the smaller slot width
        
        // This math perfectly centers the oversized ink inside the undersized slot, letting it bleed in both directions!
        let drawX = currentX + (item.slotW - item.finalW) / 2; 
        
        let yPos = 0;
        
        if (item.type === 'space') {
            if (isDebugMode) drawHitbox(ctx, currentX, 0, item.slotW, 64, '#00FF00'); 
        } 
        else if (item.char === ':') {
            let topY = baseline - (0.8 * LINE_HEIGHT_PX) - (item.finalH / 2);
            let bottomY = baseline - (item.yPosTop * LINE_HEIGHT_PX);
            drawScaledLetter(ctx, item.pixels, item.cropW, item.cropH, drawX, topY, item.scaleFactor, isColorInverted);
            drawScaledLetter(ctx, item.pixels, item.cropW, item.cropH, drawX, bottomY, item.scaleFactor, isColorInverted);
            if (isDebugMode) { drawHitbox(ctx, drawX, topY, item.finalW, item.finalH, '#FF0000'); drawHitbox(ctx, drawX, bottomY, item.finalW, item.finalH, '#FF0000'); }
        }
        else if (item.char === ';') {
            let dotScale = item.scaleFactor * 0.6; 
            let dotW = item.cropW * dotScale;
            let dotH = item.cropH * dotScale;

            let topY = baseline - (0.8 * LINE_HEIGHT_PX) - (dotH / 2);
            let bottomY = baseline - (item.yPosTop * LINE_HEIGHT_PX);
            
            let topCenterX = drawX + (item.finalW - dotW) / 2;

            drawScaledLetter(ctx, item.pixels, item.cropW, item.cropH, topCenterX, topY, dotScale, isColorInverted);
            drawScaledLetter(ctx, item.pixels, item.cropW, item.cropH, drawX, bottomY, item.scaleFactor, isColorInverted);
            
            if (isDebugMode) { 
                drawHitbox(ctx, topCenterX, topY, dotW, dotH, '#FF0000'); 
                drawHitbox(ctx, drawX, bottomY, item.finalW, item.finalH, '#FF0000'); 
            }
        }
        else if (item.char === '"') {
            yPos = baseline - (item.yPosTop * LINE_HEIGHT_PX); 
            let halfW = item.finalW / 2;
            drawScaledLetter(ctx, item.pixels, item.cropW, item.cropH, drawX + halfW, yPos, item.scaleFactor, isColorInverted);
            drawScaledLetter(ctx, item.pixels, item.cropW, item.cropH, drawX, yPos, item.scaleFactor, isColorInverted);
            if (isDebugMode) { drawHitbox(ctx, drawX + halfW, yPos, halfW, item.finalH, '#FF0000'); drawHitbox(ctx, drawX, yPos, halfW, item.finalH, '#FF0000'); }
        }
        else {
            yPos = baseline - (item.yPosTop * LINE_HEIGHT_PX);
            drawScaledLetter(ctx, item.pixels, item.cropW, item.cropH, drawX, yPos, item.scaleFactor, isColorInverted);
            if (isDebugMode) drawHitbox(ctx, drawX, yPos, item.finalW, item.finalH, '#FF0000'); 
        }

        if (isDebugMode) {
            let btn = document.createElement('button');
            btn.id = `regen-btn-${i}`;
            btn.className = 'secondary-btn'; 
            btn.style.position = 'absolute';
            btn.style.pointerEvents = 'auto'; 
            btn.style.boxSizing = 'border-box'; 
            
            let buttonY = (baseline - LINE_HEIGHT_PX) - 22; 
            
            // Buttons tile cleanly using the underlying slot width
            btn.style.left = `${currentX}px`; 
            btn.style.top = `${buttonY}px`; 
            btn.style.width = `${item.slotW}px`; 
            btn.style.height = '18px';
            btn.style.padding = '0';
            btn.style.borderRadius = '4px';
            btn.style.zIndex = '10';
            btn.style.display = 'flex';
            btn.style.justifyContent = 'center';
            btn.style.alignItems = 'center';
            
            let opacity = item.type === 'space' ? '0.4' : '1';
            btn.innerHTML = `<img src="assets/rotate-right.png" style="width: 10px; height: 10px; filter: var(--icon-filter); opacity: ${opacity};">`;
            btn.onclick = () => regenerateSingleChar(i);
            overlay.appendChild(btn);
        }
    }
    canvas.classList.add('loaded');
}

// --- UPDATED: Digital Ink Rendering ---
// --- FINAL INK RENDERING: Crisp Edges & Transparent Paper ---
function drawScaledLetter(ctx, pixelArray, rawWidth, rawHeight, xOffset, yOffset, scale, invert) {
    const offCanvas = document.createElement('canvas');
    offCanvas.width = rawWidth; 
    offCanvas.height = rawHeight;
    const offCtx = offCanvas.getContext('2d');
    
    const imgData = offCtx.createImageData(rawWidth, rawHeight);
    for (let i = 0; i < pixelArray.length; i++) {
        let val = pixelArray[i]; // 0 = Ink, 255 = Paper
        
        // If invert is OFF (Black background), flip the pixel values!
        if (!invert) val = 255 - val; 
        
        let idx = i * 4;
        imgData.data[idx] = val;     // R
        imgData.data[idx+1] = val;   // G
        imgData.data[idx+2] = val;   // B
        imgData.data[idx+3] = 255;   // Keep Alpha solid!
    }
    offCtx.putImageData(imgData, 0, 0); 
    
    // 1. Turn ON smoothing to get organic curves instead of jagged pixels
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // 2. THE CRISPNESS TRICK: Overdrive the contrast to kill the blurry smear!
    ctx.filter = 'contrast(200%)'; 
    
    // 3. THE TRANSPARENCY TRICK: Professional Blend Modes
    // 'Multiply' makes white paper invisible. 'Screen' makes black paper invisible.
    ctx.globalCompositeOperation = invert ? 'multiply' : 'screen';
    
    // Draw the crisp, transparent ink!
    ctx.drawImage(offCanvas, xOffset, yOffset, rawWidth * scale, rawHeight * scale);
    
    // 4. Reset the context so the Hitboxes and UI draw normally on top
    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'source-over';
}

// --- UPDATED: Translucent Overlapping Debug Boxes ---
function drawHitbox(ctx, x, y, width, height, color) {
    // Fill the box with 20% opacity (Hex '33') so overlapping boxes visually stack
    ctx.fillStyle = color + '33'; 
    ctx.fillRect(x, y, width, height);
    
    // Stroke the border with 80% opacity (Hex 'CC')
    ctx.strokeStyle = color + 'CC'; 
    ctx.lineWidth = 1; 
    ctx.strokeRect(x, y, width, height);
}

async function loadRandomQuote() {
    try {
        const response = await fetch('quotes.json?t=' + new Date().getTime());
        const quotes = await response.json();
        const randomItem = quotes[Math.floor(Math.random() * quotes.length)];
        document.getElementById('textInput').value = randomItem.quote;
        generateWord();
    } catch (e) { showStatus("לא הצלחנו לטעון ציטוטים", "error.png"); }
}

async function processImage() {
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files || fileInput.files.length === 0) return;

    showStatus("מנתח תמונה... (זה עשוי לקחת כמה שניות)", "exclamation.png");
    try {
        const result = await Tesseract.recognize(fileInput.files[0], 'heb');
        document.getElementById('textInput').value = result.data.text.trim();
        hideStatus(); generateWord();
    } catch (e) { showStatus("שגיאה בחילוץ הטקסט", "error.png"); }
}

const themeToggle = document.getElementById('themeToggle');
const themeIcon = themeToggle.querySelector('img');
themeToggle.addEventListener('click', () => {
    if (document.documentElement.getAttribute('data-theme') === 'dark') {
        document.documentElement.setAttribute('data-theme', 'light');
        themeIcon.src = 'assets/dark-mode-alt.png'; 
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeIcon.src = 'assets/light-mod-alt.png'; 
    }
});

document.getElementById('textInput').addEventListener('keypress', (e) => { if (e.key === 'Enter') generateWord(); });

init();