const { useState, useEffect, useRef, useCallback } = React;

const chars = 'אבגדהוזחטיכךלמםנןסעפףצץקרשת.,-!?/)( ';
const charToLabel = Object.fromEntries([...chars].map((c, i) => [c, i]));
const LINE_HEIGHT_PX = 22;

// --- Helper Math ---
function getGaussianFloat(mean, sd) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return (Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * sd) + mean;
}

function getGaussian(mean, sd) {
    return Math.round(getGaussianFloat(mean, sd));
}

function generateNoiseVector() {
    let noise = new Float32Array(100);
    for (let i = 0; i < 100; i++) {
        noise[i] = getGaussianFloat(0, 1);
    }
    return noise;
}

function hexToRgb(hex) {
    let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) } : { r: 0, g: 0, b: 0 };
}

// --- Typography Engine ---
function getLetterBounds(char) {
    let top = 1, bottom = 0, sd = 0.05;
    if ('לטצץף'.includes(char)) { top = 1.6; bottom = 0; sd = 0.1; }
    else if ('ךןק'.includes(char)) { top = 1; bottom = -0.6; sd = 0.1; }
    else if ('ת'.includes(char)) { top = 1; bottom = -0.6; sd = 0.1; }
    else if ('י'.includes(char)) { top = 1.1; bottom = 0.4; sd = 0.05; }
    else if ('\'"'.includes(char)) { top = 1.2; bottom = 0.8; sd = 0.05; }
    else if ('.,:;'.includes(char)) { top = 0.45; bottom = -0.1; sd = 0.02; }
    else if ('-'.includes(char)) { top = 0.7; bottom = 0.3; sd = 0.02; }
    else if ('/'.includes(char)) { top = 1.2; bottom = -0.2; sd = 0.05; }
    else if ('()!?'.includes(char)) { top = 1.2; bottom = -0.2; sd = 0.1; }
    return { top, bottom, sd };
}

function getKerningRatio(char) {
    if ('ל'.includes(char)) return 0.40;
    if ('ףץע'.includes(char)) return 0.25;
    if ('ויזןך'.includes(char)) return 0.05;
    if ('.,:;\'"'.includes(char)) return 0.0;
    return 0.15;
}

// --- Crop Engine (CPU fast pass just for bounding box) ---
function processAndCropRaw(flatPixelArray) {
    const width = 64, height = 64;
    let minX = width, maxX = 0, minY = height, maxY = 0;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let val = flatPixelArray[y * width + x];
            let mapped = (val + 1.0) / 2.0;
            if (mapped > 0.4) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    let isEmpty = (minX > maxX || minY > maxY);
    if (isEmpty) return { isEmpty: true, cropW: width, cropH: height, minX: 0, minY: 0 };

    minX = Math.max(0, minX - 2);
    maxX = Math.min(width - 1, maxX + 2);
    minY = Math.max(0, minY - 2);
    maxY = Math.min(height - 1, maxY + 2);

    let cropW = maxX - minX + 1;
    let cropH = maxY - minY + 1;

    let croppedPixels = new Float32Array(cropW * cropH);
    for (let cy = 0; cy < cropH; cy++) {
        for (let cx = 0; cx < cropW; cx++) {
            croppedPixels[cy * cropW + cx] = flatPixelArray[(minY + cy) * width + (minX + cx)];
        }
    }

    return { isEmpty: false, pixels: croppedPixels, cropW, cropH, minX, minY };
}

// Applies math to generate permanent layout info for a single item
function computeItemLayout(item) {
    let drawItem = { ...item };
    if (drawItem.type === 'space') {
        drawItem.finalW = drawItem.width;
        if (!drawItem.slotW) drawItem.slotW = drawItem.finalW;
    } else {
        let bounds = getLetterBounds(drawItem.char);
        let targetHeightUnits = Math.abs(bounds.top - bounds.bottom);

        // Preserve jitter so letters don't jiggle on re-renders!
        if (drawItem.heightJitter === undefined) drawItem.heightJitter = getGaussianFloat(0, bounds.sd);
        targetHeightUnits += drawItem.heightJitter;

        let targetHeightPx = targetHeightUnits * LINE_HEIGHT_PX;
        let scaleFactor = targetHeightPx / drawItem.cropH;

        drawItem.finalH = targetHeightPx;
        drawItem.finalW = drawItem.cropW * scaleFactor;
        drawItem.scaleFactor = scaleFactor;
        drawItem.yPosTop = bounds.top;

        if (drawItem.char === '"') drawItem.finalW *= 2;

        let overlapRatio = getKerningRatio(drawItem.char);
        drawItem.slotW = Math.max(5, drawItem.finalW * (1 - overlapRatio));
    }
    return drawItem;
}

// --- Main App Component ---
function App() {
    const [session, setSession] = useState(null);
    const [status, setStatus] = useState(null);
    const [text, setText] = useState("שלום עולם");
    const [generatedData, setGeneratedData] = useState(null);
    const [isModelLoaded, setIsModelLoaded] = useState(false);

    const [isDebugMode, setIsDebugMode] = useState(false);
    const [isCurveEnabled, setIsCurveEnabled] = useState(true);
    const [isDarkMode, setIsDarkMode] = useState(true);
    const [isPaperBlack, setIsPaperBlack] = useState(false);

    // Fast local state for input, debounced state for canvas draw
    const [draftTextColor, setDraftTextColor] = useState("#000000");
    const [textColor, setTextColor] = useState("#000000");
    const colorTimeoutRef = useRef(null);

    const canvasRef = useRef(null);
    const wrapperRef = useRef(null);

    // Initialize ONNX
    useEffect(() => {
        async function initModel() {
            try {
                const sess = await ort.InferenceSession.create(`./hebrew_gan_bundled_V7.2.8.onnx?v=${Date.now()}`);
                setSession(sess);
                setIsModelLoaded(true);
            } catch (e) {
                console.error(e);
                setStatus({ text: "שגיאה בטעינת המודל", type: "error" });
            }
        }
        initModel();
    }, []);

    const generateWord = useCallback(async (customText = text) => {
        if (!session) return;
        const cleanText = customText.replace(/[^א-ת.,\-!?/)( :;'"'\n]/g, '');
        if (!cleanText) {
            setStatus({ text: "אנא הזן תווים חוקיים", type: "error" });
            return;
        }

        setStatus({ text: "מייצר טקסט...", type: "info" });
        await new Promise(r => setTimeout(r, 50)); // Ensure browser paints the banner

        const items = [];
        for (let i = 0; i < cleanText.length; i++) {
            const char = cleanText[i];
            if (char === '\n') {
                items.push({ type: 'newline', slotW: 0, finalW: 0, heightJitter: 0 });
                continue;
            }
            if (char === ' ') {
                items.push(computeItemLayout({ type: 'space', width: Math.min(Math.max(7, getGaussian(20, 7)), 16), char: ' ' }));
                continue;
            }

            let modelChar = char;
            if (char === ':') modelChar = '.';
            if (char === "'" || char === '"' || char === ';') modelChar = ',';

            const labelInt = charToLabel[modelChar] !== undefined ? charToLabel[modelChar] : 0;
            const labelTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(labelInt)]), [1]);

            let validLetter = false, attempts = 0, cropInfo = null, rawPixels = null;
            while (!validLetter && attempts < 10) {
                try {
                    let noiseTensor = new ort.Tensor('float32', generateNoiseVector(), [1, 100]);
                    const results = await session.run({ noise: noiseTensor, label: labelTensor });
                    rawPixels = new Float32Array(results.generated_image.data);

                    // Post-processing fix: Boost faint disconnected strokes for problematic letters
                    if ('הקא'.includes(char)) {
                        for (let j = 0; j < rawPixels.length; j++) {
                            let mapped = (rawPixels[j] + 1.0) / 2.0;
                            mapped = Math.pow(mapped, 0.45); // Aggressive gamma boost for faint dots
                            rawPixels[j] = (mapped * 2.0) - 1.0;
                        }
                    }

                    cropInfo = processAndCropRaw(rawPixels);
                    if (cropInfo.isEmpty) attempts++; else validLetter = true;
                } catch (e) { break; }
            }

            if (cropInfo && !cropInfo.isEmpty) {
                items.push(computeItemLayout({ type: 'char', char: char, rawPixels, ...cropInfo }));
            }
        }

        setGeneratedData({
            items,
            baselineY: 64 - getGaussian(16, 2),
            id: Date.now()
        });
        setStatus(null);
    }, [session, text]);

    useEffect(() => {
        if (session && !generatedData) {
            generateWord();
        }
    }, [session]);

    const regenerateSingleChar = async (index) => {
        if (!generatedData || !session) return;
        const newItems = [...generatedData.items];
        const oldItem = newItems[index];

        if (oldItem.type === 'space') {
            let newItem = { ...oldItem, width: Math.max(5, getGaussian(25, 7)) };
            newItem.slotW = newItem.width;
            newItem.finalW = newItem.width;
            newItems[index] = newItem;
            setGeneratedData({ ...generatedData, items: newItems, id: Date.now() });
            return;
        }

        let modelChar = oldItem.char;
        if (oldItem.char === ':') modelChar = '.';
        if (oldItem.char === "'" || oldItem.char === '"' || oldItem.char === ';') modelChar = ',';

        const labelInt = charToLabel[modelChar] !== undefined ? charToLabel[modelChar] : 0;
        const labelTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(labelInt)]), [1]);

        let validLetter = false, attempts = 0, cropInfo = null, rawPixels = null;
        while (!validLetter && attempts < 10) {
            try {
                let noiseTensor = new ort.Tensor('float32', generateNoiseVector(), [1, 100]);
                const results = await session.run({ noise: noiseTensor, label: labelTensor });
                rawPixels = new Float32Array(results.generated_image.data);

                // Post-processing fix: Boost faint disconnected strokes for problematic letters
                if ('הקא'.includes(oldItem.char)) {
                    for (let j = 0; j < rawPixels.length; j++) {
                        let mapped = (rawPixels[j] + 1.0) / 2.0;
                        mapped = Math.pow(mapped, 0.45); // Aggressive gamma boost for faint dots
                        rawPixels[j] = (mapped * 2.0) - 1.0;
                    }
                }

                cropInfo = processAndCropRaw(rawPixels);
                if (cropInfo.isEmpty) attempts++; else validLetter = true;
            } catch (e) { break; }
        }

        if (cropInfo && !cropInfo.isEmpty) {
            let newItem = { ...oldItem, rawPixels, ...cropInfo };
            // Remove old properties so layout completely recalculates for the new shape!
            delete newItem.heightJitter;
            delete newItem.slotW;

            newItem = computeItemLayout(newItem);

            newItems[index] = newItem;
            setGeneratedData({ ...generatedData, items: newItems, id: Date.now() });
        }
    };

    // --- The Drawing / Rendering Logic ---
    useEffect(() => {
        if (!generatedData || !canvasRef.current || !wrapperRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        const userColorRGB = hexToRgb(textColor);

        const LINE_SPACING = isDebugMode ? 64 : 48;

        const lines = [[]];
        generatedData.items.forEach(item => {
            if (item.type === 'newline') lines.push([]);
            else lines[lines.length - 1].push(item);
        });

        let maxLineWidth = 0;
        lines.forEach(line => {
            let width = 0;
            line.forEach(item => width += item.slotW);
            if (width > maxLineWidth) maxLineWidth = width;
        });

        canvas.width = Math.max(10, maxLineWidth + 40);
        canvas.height = Math.max(64, lines.length * LINE_SPACING + 20);

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        let baseline = generatedData.baselineY;

        lines.forEach((line, lineIndex) => {
            let currentX = canvas.width - 20;
            let currentBaseline = baseline + (lineIndex * LINE_SPACING);

            if (isDebugMode) {
                ctx.strokeStyle = '#38bdf8'; ctx.beginPath(); ctx.moveTo(0, currentBaseline); ctx.lineTo(canvas.width, currentBaseline); ctx.stroke();
                ctx.strokeStyle = '#818cf8'; ctx.beginPath(); ctx.moveTo(0, currentBaseline - LINE_HEIGHT_PX); ctx.lineTo(canvas.width, currentBaseline - LINE_HEIGHT_PX); ctx.stroke();
            }

            const drawScaledLetter = (pixels, rawWidth, rawHeight, xOffset, yOffset, scale) => {
                let finalW = Math.max(1, Math.floor(rawWidth * scale));
                let finalH = Math.max(1, Math.floor(rawHeight * scale));

                // SUPER-SAMPLING FACTOR
                // We scale all raw pixels to a massive resolution first so the browser's bilinear
                // filter applies identical smooth gradients to tall and short letters alike!
                const SS_FACTOR = 3;
                const highResW = Math.floor(rawWidth * SS_FACTOR);
                const highResH = Math.floor(rawHeight * SS_FACTOR);

                const tinyCanvas = document.createElement('canvas');
                tinyCanvas.width = rawWidth;
                tinyCanvas.height = rawHeight;
                const tinyCtx = tinyCanvas.getContext('2d');
                const tinyImgData = tinyCtx.createImageData(rawWidth, rawHeight);

                for (let i = 0; i < pixels.length; i++) {
                    let x = (pixels[i] + 1.0) / 2.0;
                    x = Math.max(0, Math.min(1, x));
                    let grayscale = Math.floor((1.0 - x) * 255);
                    let idx = i * 4;
                    tinyImgData.data[idx] = grayscale;
                    tinyImgData.data[idx + 1] = grayscale;
                    tinyImgData.data[idx + 2] = grayscale;
                    tinyImgData.data[idx + 3] = 255;
                }
                tinyCtx.putImageData(tinyImgData, 0, 0);

                // Draw to High-Res Canvas WITH Smoothing (Forces identical blur on all letters)
                const highResCanvas = document.createElement('canvas');
                highResCanvas.width = highResW;
                highResCanvas.height = highResH;
                const highResCtx = highResCanvas.getContext('2d', { willReadFrequently: true });
                highResCtx.imageSmoothingEnabled = true;
                highResCtx.imageSmoothingQuality = 'high';
                highResCtx.drawImage(tinyCanvas, 0, 0, highResW, highResH);

                const midImgData = highResCtx.getImageData(0, 0, highResW, highResH);
                for (let i = 0; i < midImgData.data.length; i += 4) {
                    let grayscale = midImgData.data[i] / 255.0;
                    let density = 1.0 - grayscale;

                    let alpha = 0;
                    if (isCurveEnabled) {
                        let val = (density * 2.0) - 1.0;
                        let clamped = (val + 0.1) / 0.8;
                        clamped = Math.max(0, Math.min(1, clamped));
                        let inkDensity = Math.pow(clamped, 2.5);
                        alpha = Math.floor(inkDensity * 255);
                    } else {
                        alpha = Math.floor(density * 255);
                    }

                    midImgData.data[i] = userColorRGB.r;
                    midImgData.data[i + 1] = userColorRGB.g;
                    midImgData.data[i + 2] = userColorRGB.b;
                    midImgData.data[i + 3] = alpha;
                }
                highResCtx.putImageData(midImgData, 0, 0);

                ctx.globalCompositeOperation = 'source-over';
                ctx.imageSmoothingEnabled = true;
                ctx.drawImage(highResCanvas, xOffset, yOffset, finalW, finalH);
                ctx.globalCompositeOperation = 'source-over';
            };

            const drawHitbox = (x, y, w, h, color) => {
                ctx.fillStyle = color + '33';
                ctx.fillRect(x, y, w, h);
                ctx.strokeStyle = color + 'CC';
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, w, h);
            };

            line.forEach((item) => {
                currentX -= item.slotW;
                let drawX = currentX + (item.slotW - item.finalW) / 2;
                let yPos = 0;

                if (item.type === 'space') {
                    if (isDebugMode) drawHitbox(currentX, currentBaseline - LINE_HEIGHT_PX, item.slotW, LINE_HEIGHT_PX, '#10b981');
                }
                else if (item.char === ':') {
                    let topY = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (item.finalH / 2);
                    let bottomY = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
                    drawScaledLetter(item.pixels, item.cropW, item.cropH, drawX, topY, item.scaleFactor);
                    drawScaledLetter(item.pixels, item.cropW, item.cropH, drawX, bottomY, item.scaleFactor);
                    if (isDebugMode) { drawHitbox(drawX, topY, item.finalW, item.finalH, '#ef4444'); drawHitbox(drawX, bottomY, item.finalW, item.finalH, '#ef4444'); }
                }
                else if (item.char === ';') {
                    let dotScale = item.scaleFactor * 0.6;
                    let dotW = item.cropW * dotScale;
                    let dotH = item.cropH * dotScale;
                    let topY = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (dotH / 2);
                    let bottomY = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
                    let topCenterX = drawX + (item.finalW - dotW) / 2;
                    drawScaledLetter(item.pixels, item.cropW, item.cropH, topCenterX, topY, dotScale);
                    drawScaledLetter(item.pixels, item.cropW, item.cropH, drawX, bottomY, item.scaleFactor);
                    if (isDebugMode) { drawHitbox(topCenterX, topY, dotW, dotH, '#ef4444'); drawHitbox(drawX, bottomY, item.finalW, item.finalH, '#ef4444'); }
                }
                else if (item.char === '"') {
                    yPos = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
                    let halfW = item.finalW / 2;
                    drawScaledLetter(item.pixels, item.cropW, item.cropH, drawX + halfW, yPos, item.scaleFactor);
                    drawScaledLetter(item.pixels, item.cropW, item.cropH, drawX, yPos, item.scaleFactor);
                    if (isDebugMode) { drawHitbox(drawX + halfW, yPos, halfW, item.finalH, '#ef4444'); drawHitbox(drawX, yPos, halfW, item.finalH, '#ef4444'); }
                }
                else {
                    yPos = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
                    drawScaledLetter(item.pixels, item.cropW, item.cropH, drawX, yPos, item.scaleFactor);
                    if (isDebugMode) drawHitbox(drawX, yPos, item.finalW, item.finalH, '#ef4444');
                }
            });
        });

    }, [generatedData, isDebugMode, isCurveEnabled, isPaperBlack, textColor]);


    // Handlers
    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey || e.shiftKey)) {
            e.preventDefault();
            generateWord();
        }
    };

    const loadRandomQuote = async () => {
        try {
            const response = await fetch('quotes.json?t=' + new Date().getTime());
            const quotes = await response.json();
            const randomItem = quotes[Math.floor(Math.random() * quotes.length)];
            const formattedText = `"${randomItem.quote}"\n- ${randomItem.author}`;
            setText(formattedText);
            generateWord(formattedText);
        } catch (e) {
            setStatus({ text: "לא הצלחנו לטעון ציטוטים", type: "error" });
        }
    };

    const processImage = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setStatus({ text: "מנתח תמונה... (זה עשוי לקחת כמה שניות)", type: "info" });
        try {
            const result = await Tesseract.recognize(file, 'heb');
            const extractedText = result.data.text.trim();
            setText(extractedText);
            generateWord(extractedText);
        } catch (err) {
            setStatus({ text: "שגיאה בחילוץ הטקסט", type: "error" });
        }
    };

    const toggleTheme = () => {
        const newMode = !isDarkMode;
        setIsDarkMode(newMode);
        document.documentElement.setAttribute('data-theme', newMode ? 'dark' : 'light');
    };

    const togglePaperColor = () => {
        const newBlack = !isPaperBlack;
        setIsPaperBlack(newBlack);
        if (newBlack && textColor === '#000000') {
            setDraftTextColor('#ffffff');
            setTextColor('#ffffff');
        }
        if (!newBlack && textColor === '#ffffff') {
            setDraftTextColor('#000000');
            setTextColor('#000000');
        }
    };

    const handleColorChange = (e) => {
        const newColor = e.target.value;
        setDraftTextColor(newColor);
        if (colorTimeoutRef.current) clearTimeout(colorTimeoutRef.current);
        colorTimeoutRef.current = setTimeout(() => {
            setTextColor(newColor);
        }, 80);
    };

    const copyToClipboard = () => {
        if (!canvasRef.current) return;
        canvasRef.current.toBlob((blob) => {
            try {
                navigator.clipboard.write([
                    new ClipboardItem({ 'image/png': blob })
                ]);
                setStatus({ text: "התמונה הועתקה ללוח!", type: "success" });
                setTimeout(() => setStatus(null), 2500);
            } catch (e) {
                setStatus({ text: "שגיאה בהעתקת התמונה", type: "error" });
            }
        });
    };

    return (
        <div className="app-container">
            <div className={`loading-overlay ${isModelLoaded ? 'fade-out' : ''}`}>
                <div className="spinner"></div>
                <div className="loading-text">טוען מודל...</div>
            </div>

            <div className="bg-orb orb-1"></div>
            <div className="bg-orb orb-2"></div>

            <header className="glass-header">
                <div className="header-content">
                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '15px' }}>
                        <h1 style={{ margin: 0 }}>מחולל כתב יד <span>בזמן אמת</span></h1>
                        <button className="tool-btn theme-toggle" onClick={toggleTheme} title="שנה מצב תצוגה (UI)" style={{ border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                            <img src={isDarkMode ? "assets/sun.png" : "assets/moon.png"} alt="Toggle Theme" className="icon-img" />
                        </button>
                    </div>
                    <p>מודל WGAN-GP הרץ ישירות בדפדפן - כחלק מפרויקט למידת מכונה - יואב גראד</p>
                </div>
            </header>

            <main className="main-content">
                <section className="glass-card controls-card">
                    <h2>מה תרצה לכתוב?</h2>
                    <div className="input-row">
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            onKeyDown={handleKeyPress}
                            placeholder="הקלד כאן... (אנטר לשורה חדשה, Ctrl+Enter ליצירה)"
                            className="text-input"
                            style={{ resize: 'vertical', minHeight: '60px', flex: 1, fontFamily: 'inherit', fontSize: '1.1rem', padding: '10px' }}
                        />
                        <button className="btn-primary" onClick={() => generateWord()} title="צייר כתב יד">
                            <img src="assets/write.png" alt="Write" className="icon-img" />
                        </button>
                    </div>

                    <div className="toolbar">
                        <label className="tool-btn" title="טען תמונה ל-OCR">
                            <input type="file" accept="image/*" style={{ display: 'none' }} onChange={processImage} />
                            <img src="assets/picture.png" alt="OCR" className="icon-img" />
                        </label>

                        <button className="tool-btn" onClick={loadRandomQuote} title="ציטוט אקראי">
                            <img src="assets/quote-right.png" alt="Quote" className="icon-img" />
                        </button>

                        <div className="divider"></div>

                        <button className="tool-btn" onClick={copyToClipboard} title="העתק תמונה ללוח">
                            <img src="assets/copy.png" alt="Copy Image" className="icon-img" />
                        </button>

                        <div className="divider"></div>

                        <button className={`tool-btn ${isDebugMode ? 'active' : ''}`} onClick={() => setIsDebugMode(!isDebugMode)} title="מצב דיבאג (מיקומי אותיות)">
                            <img src="assets/debug.png" alt="Debug" className="icon-img" />
                        </button>

                        <button className={`tool-btn ${isCurveEnabled ? 'active' : ''}`} onClick={() => setIsCurveEnabled(!isCurveEnabled)} title="אפקט דיו רטוב (חדות)">
                            <img src="assets/curve.png" alt="Curve" className="icon-img" />
                        </button>

                        <button className={`tool-btn ${isPaperBlack ? 'active' : ''}`} onClick={togglePaperColor} title="שנה צבע נייר (שחור/לבן)">
                            <img src="assets/invert.png" alt="Invert Paper" className="icon-img" />
                        </button>

                        <div className="color-picker-wrapper tool-btn" title="בחר צבע דיו">
                            <input
                                type="color"
                                value={draftTextColor}
                                onChange={handleColorChange}
                                className="color-wheel"
                            />
                        </div>

                        <button className="tool-btn" onClick={() => generateWord()} title="הגרל מחדש את כל המילה">
                            <img src="assets/rotate-right.png" alt="Refresh" className="icon-img" />
                        </button>
                    </div>

                    {status && (
                        <div className={`status-pill ${status.type}`}>
                            {status.type === 'error' && <img src="assets/error.png" alt="Error" width="14" height="14" style={{ filter: 'var(--icon-filter)' }} />}
                            {status.type === 'info' && <img src="assets/exclamation.png" alt="Info" width="14" height="14" style={{ filter: 'var(--icon-filter)' }} />}
                            <span>{status.text}</span>
                        </div>
                    )}
                </section>

                <section className="glass-card canvas-card">
                    <h2>תוצאה</h2>
                    <div className="canvas-overflow-wrapper">
                        <div className="canvas-inner-wrapper" ref={wrapperRef}>
                            <canvas ref={canvasRef} className={`main-canvas ${isPaperBlack ? 'paper-black' : 'paper-white'}`} />
                            {isDebugMode && generatedData && (
                                <div className="debug-buttons-layer">
                                    {(() => {
                                        const LINE_SPACING = isDebugMode ? 65 : 45;
                                        let currRight = 20;
                                        let currTop = generatedData.baselineY - 55;
                                        return generatedData.items.map((item, i) => {
                                            if (item.type === 'newline') {
                                                currRight = 20;
                                                currTop += LINE_SPACING;
                                                return null;
                                            }
                                            let myRight = currRight;
                                            currRight += item.slotW;
                                            return (
                                                <button
                                                    key={`regen-${i}-${item.id}`}
                                                    className="regen-char-btn"
                                                    style={{
                                                        right: `${myRight}px`,
                                                        top: `${currTop}px`,
                                                        width: `${item.slotW}px`
                                                    }}
                                                    onClick={() => regenerateSingleChar(i)}
                                                    title="הגרל אות זו מחדש"
                                                >
                                                    <img src="assets/rotate-right.png" alt="Regen" style={{ width: '10px', height: '10px', filter: 'var(--icon-filter)' }} />
                                                </button>
                                            );
                                        });
                                    })()}
                                </div>
                            )}
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
