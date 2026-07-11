const { useState, useEffect, useRef, useCallback } = React;

const chars = 'אבגדהוזחטיכךלמםנןסעפףצץקרשת.,-!?/)( ';
const charToLabel = Object.fromEntries([...chars].map((c, i) => [c, i]));
const LINE_HEIGHT_PX = 22;
const ONNX_RUNTIME_VERSION = '1.27.0';
const ONNX_WASM_BASE_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_RUNTIME_VERSION}/dist/`;
const MODELS = [
    { id: 'v7.2.8', name: 'Model V7.2.8', url: './hebrew_gan_int8_V7.2.8.onnx' },
    { id: 'v7.4.1-fid', name: 'Model V7.4.1 FID', url: './hebrew_gan_int8_V7.4.1-FID.onnx' },
    { id: 'v7.4.2', name: 'Model V7.4.2', url: './hebrew_gan_int8_V7.4.2.onnx' }
];
const DEFAULT_MODEL_ID = 'v7.2.8';
const UI_TEXT = {
    he: {
        pageTitle: 'מחולל כתב יד בזמן אמת',
        header: 'מחולל כתב יד', accent: 'בזמן אמת',
        subtitle: 'מודל WGAN-GP הרץ ישירות בדפדפן - כחלק מפרויקט למידת מכונה - יואב גראד',
        loadingModel: 'טוען מודל...', loadingSelected: model => `טוען ${model}...`,
        loadingFile: 'טוען את קובץ המודל...', validating: 'בודק את מנוע הכתיבה...', loadingFailed: 'טעינת המודל נכשלה',
        modelReady: 'המודל נטען ונבדק בהצלחה!', modelLoadError: error => `שגיאה בטעינת המודל: ${error}`,
        validCharacters: 'אנא הזן תווים חוקיים', generating: 'מייצר טקסט...',
        generationError: error => `שגיאה ביצירת כתב היד: ${error}`, regenerateError: error => `שגיאה ביצירת האות מחדש: ${error}`,
        quoteLoadError: 'לא הצלחנו לטעון ציטוטים', ocrProcessing: 'מנתח תמונה... (זה עשוי לקחת כמה שניות)',
        ocrError: 'שגיאה בחילוץ הטקסט', copied: 'התמונה הועתקה ללוח!', copyError: 'שגיאה בהעתקת התמונה',
        changeTheme: 'שנה מצב תצוגה', prompt: 'מה תרצה לכתוב?',
        inputPlaceholder: 'הקלד כאן... (אנטר לשורה חדשה, Ctrl+Enter ליצירה)', write: 'צייר כתב יד',
        uploadOcr: 'טען תמונה ל-OCR', randomQuote: 'ציטוט אקראי', selectModel: 'בחר מודל',
        copy: 'העתק תמונה ללוח', debug: 'מצב דיבאג (מיקומי אותיות)', quality: 'בקרת איכות תווים',
        paper: 'שנה צבע נייר (שחור/לבן)', inkColor: 'בחר צבע דיו', refresh: 'הגרל מחדש את כל המילה',
        result: 'תוצאה', regenerate: 'הגרל אות זו מחדש', language: 'English', languageLabel: 'שפת ממשק',
        themeIcon: 'שנה תצוגה', writeIcon: 'כתיבה', ocrIcon: 'OCR', quoteIcon: 'ציטוט', copyIcon: 'העתקה',
        debugIcon: 'דיבאג', qualityIcon: 'איכות', paperIcon: 'נייר', refreshIcon: 'רענון', errorIcon: 'שגיאה', infoIcon: 'מידע'
    },
    en: {
        pageTitle: 'Real-Time Handwriting Generator',
        header: 'Handwriting Generator', accent: 'in Real Time',
        subtitle: 'A WGAN-GP model running directly in your browser — a machine-learning project by Yoav Grad',
        loadingModel: 'Loading model...', loadingSelected: model => `Loading ${model}...`,
        loadingFile: 'Loading model file...', validating: 'Checking handwriting engine...', loadingFailed: 'Model loading failed',
        modelReady: 'Model loaded and verified!', modelLoadError: error => `Model loading error: ${error}`,
        validCharacters: 'Please enter valid characters', generating: 'Generating text...',
        generationError: error => `Handwriting generation error: ${error}`, regenerateError: error => `Character regeneration error: ${error}`,
        quoteLoadError: 'Could not load quotes', ocrProcessing: 'Reading image… this may take a few seconds',
        ocrError: 'Text extraction failed', copied: 'Image copied to clipboard!', copyError: 'Could not copy image',
        changeTheme: 'Change appearance', prompt: 'What would you like to write?',
        inputPlaceholder: 'Type here… (new line with Enter, generate with Ctrl+Enter)', write: 'Generate handwriting',
        uploadOcr: 'Upload image for OCR', randomQuote: 'Random quote', selectModel: 'Select model',
        copy: 'Copy image to clipboard', debug: 'Debug letter placement', quality: 'Character quality control',
        paper: 'Toggle paper color', inkColor: 'Choose ink color', refresh: 'Regenerate all text',
        result: 'Result', regenerate: 'Regenerate this character', language: 'עברית', languageLabel: 'Interface language',
        themeIcon: 'Toggle theme', writeIcon: 'Write', ocrIcon: 'OCR', quoteIcon: 'Quote', copyIcon: 'Copy image',
        debugIcon: 'Debug', qualityIcon: 'Quality control', paperIcon: 'Paper color', refreshIcon: 'Refresh', errorIcon: 'Error', infoIcon: 'Info'
    }
};
const TESSERACT_URL = 'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js';
let tesseractLoadPromise = null;

function getErrorMessage(error) {
    if (error instanceof Error && error.message) return error.message;
    return String(error);
}

function getSavedLanguage() {
    try {
        const savedLanguage = localStorage.getItem('handwriting-ui-language');
        return savedLanguage === 'en' ? 'en' : 'he';
    } catch {
        return 'he';
    }
}

function loadTesseract() {
    if (window.Tesseract) return Promise.resolve(window.Tesseract);
    if (tesseractLoadPromise) return tesseractLoadPromise;

    tesseractLoadPromise = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = TESSERACT_URL;
        script.async = true;
        script.onload = () => {
            if (window.Tesseract) resolve(window.Tesseract);
            else {
                tesseractLoadPromise = null;
                reject(new Error('The OCR engine downloaded but did not initialize.'));
            }
        };
        script.onerror = () => {
            tesseractLoadPromise = null;
            reject(new Error('Could not download the OCR engine.'));
        };
        document.head.appendChild(script);
    });

    return tesseractLoadPromise;
}

async function validateInferenceSession(session) {
    const noise = new ort.Tensor('float32', new Float32Array(100), [1, 100]);
    const label = new ort.Tensor('int64', new BigInt64Array([0n]), [1]);
    const results = await session.run({ noise, label });
    const output = results.generated_image;

    if (!output || !output.data || output.data.length !== 64 * 64) {
        const actualLength = output?.data?.length ?? 0;
        throw new Error(`Startup inference returned ${actualLength} pixels; expected 4096.`);
    }
}

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
function getLetterBounds(char, nextChar) {
    let top = 1, bottom = 0, sd = 0.05;
    if ('לטצץף'.includes(char)) { top = 1.6; bottom = 0; sd = 0.1; }
    else if ('ךןק'.includes(char)) { top = 1; bottom = -0.6; sd = 0.1; }
    else if ('ת'.includes(char)) {
        top = 1;
        bottom = -0.6;
        sd = 0.1;
        if (nextChar && !'י\'".,:;-/()!? \n'.includes(nextChar)) {
            bottom = -0.2;
        }
    }
    else if ('נ'.includes(char)) { top = 1; bottom = -0.2; sd = 0.05; }
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
    if ('ףץעךט'.includes(char)) return 0.25;
    if ('וין'.includes(char)) return 0.05;
    if ('.,:;\'"'.includes(char)) return 0.02;
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

function isValidLetter(char, cropInfo, isQualityCheckEnabled) {
    if (!isQualityCheckEnabled) return true;
    if (cropInfo.isEmpty) return false;

    let w = cropInfo.cropW;
    let h = cropInfo.cropH;
    let pixels = cropInfo.pixels;
    let isPuncOrYud = '.,:;\'"?!-/י'.includes(char);
    let isPunc = '.,:;\'"?!-/()'.includes(char);

    // Size constraint
    if (!isPuncOrYud && (w < 5 || h < 5)) return false;

    // Ink Density, Gray Pixels, Edge Pixels, and Visited Map
    let inkCount = 0;
    let grayCount = 0;
    let perimeterCount = 0;

    let isInk = new Uint8Array(w * h);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let idx = y * w + x;
            let mapped = (pixels[idx] + 1.0) / 2.0;

            if (mapped > 0.4) {
                inkCount++;
                isInk[idx] = 1;
            }
            if (mapped > 0.25 && mapped < 0.75) {
                grayCount++;
            }
        }
    }

    if (inkCount === 0) return false;

    let density = inkCount / (w * h);
    if (!isPuncOrYud && density > 0.5) return false; // Blobs
    if (density < 0.05) return false; // Noise

    // Aspect Ratio
    let ratio = w / h;
    if ('וןך'.includes(char) && ratio > 1.2) return false; // Too wide
    if ('שמת'.includes(char) && ratio < 0.6) return false; // Too narrow

    // Blurriness Check
    if (grayCount / inkCount > 0.35) return false; // Too smudgy/blurry

    // Perimeter & Thickness Check
    if (!isPunc) {
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let idx = y * w + x;
                if (isInk[idx]) {
                    let isEdge = false;
                    if (y === 0 || !isInk[(y - 1) * w + x]) isEdge = true;
                    else if (y === h - 1 || !isInk[(y + 1) * w + x]) isEdge = true;
                    else if (x === 0 || !isInk[y * w + (x - 1)]) isEdge = true;
                    else if (x === w - 1 || !isInk[y * w + (x + 1)]) isEdge = true;

                    if (isEdge) perimeterCount++;
                }
            }
        }

        if (perimeterCount > 0) {
            let thickness = (2.0 * inkCount) / perimeterCount;
            if (thickness < 1.3) return false; // Allowed thinner
            if (thickness > 4.8) return false; // Lowered thick limit
        }
    }

    // Island Counting (Flood Fill)
    let visited = new Uint8Array(w * h);
    let islands = 0;
    let queue = new Int32Array(w * h); // pre-allocate

    for (let i = 0; i < w * h; i++) {
        if (isInk[i] && !visited[i]) {
            islands++;
            visited[i] = 1;

            let head = 0, tail = 0;
            queue[tail++] = i;

            while (head < tail) {
                let curr = queue[head++];
                let cx = curr % w;
                let cy = Math.floor(curr / w);

                // Neighbors: up, down, left, right, and diagonals
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            let nidx = ny * w + nx;
                            if (isInk[nidx] && !visited[nidx]) {
                                visited[nidx] = 1;
                                queue[tail++] = nidx;
                            }
                        }
                    }
                }
            }
        }
    }

    // Island constraints
    if ('?!:'.includes(char)) {
        if (islands !== 2) return false;
    } else if ('הק'.includes(char)) {
        if (islands > 2) return false;
    } else if ('א'.includes(char)) {
        if (islands > 3) return false;
    } else if (!isPunc) {
        if (islands > 1) return false;
    }

    // Hole Counting (Background Flood Fill)
    let outsideVisited = new Uint8Array(w * h);
    let bgQueue = new Int32Array(w * h);
    let bgHead = 0, bgTail = 0;

    // 1. Flood-fill outside background from borders
    for (let x = 0; x < w; x++) {
        if (!isInk[x]) { outsideVisited[x] = 1; bgQueue[bgTail++] = x; } // top
        let bottom = (h - 1) * w + x;
        if (!isInk[bottom]) { outsideVisited[bottom] = 1; bgQueue[bgTail++] = bottom; } // bottom
    }
    for (let y = 0; y < h; y++) {
        let left = y * w;
        if (!isInk[left] && !outsideVisited[left]) { outsideVisited[left] = 1; bgQueue[bgTail++] = left; } // left
        let right = y * w + (w - 1);
        if (!isInk[right] && !outsideVisited[right]) { outsideVisited[right] = 1; bgQueue[bgTail++] = right; } // right
    }

    while (bgHead < bgTail) {
        let curr = bgQueue[bgHead++];
        let cx = curr % w;
        let cy = Math.floor(curr / w);

        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                if (dx !== 0 && dy !== 0) continue; // 4-connectivity prevents leaking through thin gaps

                let nx = cx + dx;
                let ny = cy + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    let nidx = ny * w + nx;
                    if (!isInk[nidx] && !outsideVisited[nidx]) {
                        outsideVisited[nidx] = 1;
                        bgQueue[bgTail++] = nidx;
                    }
                }
            }
        }
    }

    // 2. Count internal holes
    let holes = 0;
    for (let i = 0; i < w * h; i++) {
        if (!isInk[i] && !outsideVisited[i]) {
            holes++;
            outsideVisited[i] = 1;

            let hHead = 0, hTail = 0;
            bgQueue[hTail++] = i;
            while (hHead < hTail) {
                let curr = bgQueue[hHead++];
                let cx = curr % w;
                let cy = Math.floor(curr / w);

                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        if (dx !== 0 && dy !== 0) continue; // 4-connectivity

                        let nx = cx + dx;
                        let ny = cy + dy;
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            let nidx = ny * w + nx;
                            if (!isInk[nidx] && !outsideVisited[nidx]) {
                                outsideVisited[nidx] = 1;
                                bgQueue[hTail++] = nidx;
                            }
                        }
                    }
                }
            }
        }
    }

    // Hole constraints
    if ('סם'.includes(char)) {
        if (holes !== 1) return false; // Must have exactly 1 loop
    } else if ('טפצלתב'.includes(char)) {
        if (holes > 1) return false; // Can have 0 or 1 loop depending on cursive style
    } else if ('הקא'.includes(char)) {
        // Let it be, multi-strokes can overlap
    } else {
        if (holes > 0) return false; // Everything else shouldn't have loops
    }

    return true;
}

// Applies math to generate permanent layout info for a single item
function computeItemLayout(item) {
    let drawItem = { ...item };
    if (drawItem.type === 'space') {
        drawItem.finalW = drawItem.width;
        if (!drawItem.slotW) drawItem.slotW = drawItem.finalW;
    } else {
        let bounds = getLetterBounds(drawItem.char, drawItem.nextChar);
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

        if (drawItem.pixels) {
            let finalH_int = Math.max(1, Math.floor(drawItem.finalH));
            let leftContour = new Float32Array(finalH_int).fill(drawItem.finalW);
            let rightContour = new Float32Array(finalH_int).fill(0);

            let hasStrokes = false;
            for (let cy = 0; cy < drawItem.cropH; cy++) {
                let mappedY = Math.floor(cy * drawItem.scaleFactor);
                if (mappedY >= finalH_int) mappedY = finalH_int - 1;

                let rowMinX = drawItem.cropW;
                let rowMaxX = -1;

                for (let cx = 0; cx < drawItem.cropW; cx++) {
                    let val = drawItem.pixels[cy * drawItem.cropW + cx];
                    let mapped = (val + 1.0) / 2.0;
                    if (mapped > 0.4) {
                        if (cx < rowMinX) rowMinX = cx;
                        if (cx > rowMaxX) rowMaxX = cx;
                    }
                }

                if (rowMinX <= rowMaxX) {
                    hasStrokes = true;
                    let finalMinX = rowMinX * drawItem.scaleFactor;
                    let finalMaxX = rowMaxX * drawItem.scaleFactor;
                    if (finalMinX < leftContour[mappedY]) leftContour[mappedY] = finalMinX;
                    if (finalMaxX > rightContour[mappedY]) rightContour[mappedY] = finalMaxX;
                }
            }

            if (hasStrokes) {
                let lastLeft = drawItem.finalW, lastRight = 0;
                for (let y = 0; y < finalH_int; y++) {
                    if (leftContour[y] === drawItem.finalW) leftContour[y] = lastLeft;
                    else lastLeft = leftContour[y];

                    if (rightContour[y] === 0) rightContour[y] = lastRight;
                    else lastRight = rightContour[y];
                }
                lastLeft = drawItem.finalW; lastRight = 0;
                for (let y = finalH_int - 1; y >= 0; y--) {
                    if (leftContour[y] === drawItem.finalW) leftContour[y] = lastLeft;
                    else lastLeft = leftContour[y];

                    if (rightContour[y] === 0) rightContour[y] = lastRight;
                    else lastRight = rightContour[y];
                }
            }

            drawItem.leftContour = leftContour;
            drawItem.rightContour = rightContour;
        }

        if (drawItem.spacingJitter === undefined) {
            drawItem.spacingJitter = getGaussianFloat(3, 1);
        }
    }
    return drawItem;
}

// --- Kerning Helpers ---
function getItemYRange(item, currentBaseline) {
    let standardY = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
    let top = standardY;
    let bottom = standardY + item.finalH;

    if (item.char === ':') {
        top = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (item.finalH / 2);
    } else if (item.char === ';') {
        let dotScale = item.scaleFactor * 0.6;
        let dotH = item.cropH * dotScale;
        top = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (dotH / 2);
    }

    return { top, bottom };
}

function getRightItemLeftBound(item, absoluteY, standardYPos, currentBaseline) {
    if (!item.leftContour) return 0;
    let yLocal = absoluteY - Math.floor(standardYPos);

    if (item.char === ':') {
        let topY = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (item.finalH / 2);
        let bottomY = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
        let topLocal = absoluteY - Math.floor(topY);
        let bottomLocal = absoluteY - Math.floor(bottomY);
        let bound = item.finalW;
        if (topLocal >= 0 && topLocal < item.leftContour.length) bound = Math.min(bound, item.leftContour[topLocal]);
        if (bottomLocal >= 0 && bottomLocal < item.leftContour.length) bound = Math.min(bound, item.leftContour[bottomLocal]);
        return bound;
    }
    if (item.char === ';') {
        let dotScale = item.scaleFactor * 0.6;
        let dotH = item.cropH * dotScale;
        let topY = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (dotH / 2);
        let bottomY = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
        let topLocal = Math.floor((absoluteY - Math.floor(topY)) * (item.scaleFactor / dotScale));
        let bottomLocal = absoluteY - Math.floor(bottomY);

        let bound = item.finalW;
        if (topLocal >= 0 && topLocal < item.leftContour.length) {
            let offset = (item.finalW - (item.cropW * dotScale)) / 2;
            bound = Math.min(bound, item.leftContour[topLocal] * (dotScale / item.scaleFactor) + offset);
        }
        if (bottomLocal >= 0 && bottomLocal < item.leftContour.length) bound = Math.min(bound, item.leftContour[bottomLocal]);
        return bound;
    }
    if (item.char === '"') {
        if (yLocal >= 0 && yLocal < item.leftContour.length) return item.leftContour[yLocal];
        return item.finalW;
    }

    if (yLocal >= 0 && yLocal < item.leftContour.length) return item.leftContour[yLocal];
    return item.finalW;
}

function getLeftItemRightBound(item, absoluteY, standardYPos, currentBaseline) {
    if (!item.rightContour) return item.finalW;
    let yLocal = absoluteY - Math.floor(standardYPos);

    if (item.char === ':') {
        let topY = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (item.finalH / 2);
        let bottomY = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
        let topLocal = absoluteY - Math.floor(topY);
        let bottomLocal = absoluteY - Math.floor(bottomY);
        let bound = 0;
        if (topLocal >= 0 && topLocal < item.rightContour.length) bound = Math.max(bound, item.rightContour[topLocal]);
        if (bottomLocal >= 0 && bottomLocal < item.rightContour.length) bound = Math.max(bound, item.rightContour[bottomLocal]);
        return bound;
    }
    if (item.char === ';') {
        let dotScale = item.scaleFactor * 0.6;
        let dotH = item.cropH * dotScale;
        let topY = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (dotH / 2);
        let bottomY = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
        let topLocal = Math.floor((absoluteY - Math.floor(topY)) * (item.scaleFactor / dotScale));
        let bottomLocal = absoluteY - Math.floor(bottomY);

        let bound = 0;
        if (topLocal >= 0 && topLocal < item.rightContour.length) {
            let offset = (item.finalW - (item.cropW * dotScale)) / 2;
            bound = Math.max(bound, item.rightContour[topLocal] * (dotScale / item.scaleFactor) + offset);
        }
        if (bottomLocal >= 0 && bottomLocal < item.rightContour.length) bound = Math.max(bound, item.rightContour[bottomLocal]);
        return bound;
    }
    if (item.char === '"') {
        if (yLocal >= 0 && yLocal < item.rightContour.length) return item.rightContour[yLocal] + item.finalW / 2;
        return 0;
    }

    if (yLocal >= 0 && yLocal < item.rightContour.length) return item.rightContour[yLocal];
    return 0;
}

// --- Main App Component ---
function App() {
    const [session, setSession] = useState(null);
    const [selectedModelId, setSelectedModelId] = useState(DEFAULT_MODEL_ID);
    const [language, setLanguage] = useState(getSavedLanguage);
    const [status, setStatus] = useState(null);
    const [text, setText] = useState("שלום עולם");
    const [generatedData, setGeneratedData] = useState(null);
    const [isModelLoaded, setIsModelLoaded] = useState(false);
    const [loadingStage, setLoadingStage] = useState(UI_TEXT.he.loadingModel);

    const [isDebugMode, setIsDebugMode] = useState(false);
    const [isCurveEnabled, setIsCurveEnabled] = useState(true);
    // Disable quality check by default to make generation instantaneous (1 inference per letter)
    const [isQualityCheckEnabled, setIsQualityCheckEnabled] = useState(false);
    const [isDarkMode, setIsDarkMode] = useState(true);
    const [isPaperBlack, setIsPaperBlack] = useState(false);

    // Fast local state for input, debounced state for canvas draw
    const [draftTextColor, setDraftTextColor] = useState("#000000");
    const [textColor, setTextColor] = useState("#000000");
    const colorTimeoutRef = useRef(null);

    const canvasRef = useRef(null);
    const wrapperRef = useRef(null);
    const charCacheRef = useRef(new WeakMap());
    const kerningCacheRef = useRef(new WeakMap());
    const selectedModel = MODELS.find(model => model.id === selectedModelId) || MODELS[0];
    const t = UI_TEXT[language];

    useEffect(() => {
        document.documentElement.lang = language;
        document.documentElement.dir = language === 'he' ? 'rtl' : 'ltr';
        document.title = t.pageTitle;
        try {
            localStorage.setItem('handwriting-ui-language', language);
        } catch {
            // The UI still works when browser storage is unavailable.
        }
        setStatus(null);
        if (document.documentElement.dataset.modelStatus === 'loading') {
            setLoadingStage(t.loadingModel);
        }
    }, [language]);

    useEffect(() => {
        charCacheRef.current = new WeakMap();
    }, [textColor, isCurveEnabled]);

    // Initialize ONNX
    useEffect(() => {
        let cancelled = false;
        let loadedSession = null;

        async function initModel() {
            setSession(null);
            setGeneratedData(null);
            setIsModelLoaded(false);
            setLoadingStage(t.loadingSelected(selectedModel.name));
            document.documentElement.dataset.modelStatus = 'loading';
            document.documentElement.dataset.modelId = selectedModel.id;
            try {
                // Keep the JS runtime and its dynamically loaded WASM files on the same exact release.
                ort.env.wasm.wasmPaths = ONNX_WASM_BASE_URL;
                ort.env.wasm.numThreads = 1;

                const options = {
                    executionProviders: ['wasm']
                };
                const detectedRuntimeVersion = ort.env?.versions?.web || ONNX_RUNTIME_VERSION;
                document.documentElement.dataset.runtimeVersion = detectedRuntimeVersion;
                console.info('Loading handwriting model', {
                    runtimeVersion: detectedRuntimeVersion,
                    wasmBaseUrl: ONNX_WASM_BASE_URL,
                    modelUrl: selectedModel.url,
                    modelName: selectedModel.name,
                    executionProvider: 'wasm',
                    wasmThreads: ort.env.wasm.numThreads
                });

                setLoadingStage(t.loadingFile);
                const sess = await ort.InferenceSession.create(selectedModel.url, options);
                loadedSession = sess;
                setLoadingStage(t.validating);
                await validateInferenceSession(sess);

                if (cancelled) {
                    if (typeof sess.release === 'function') await sess.release();
                    loadedSession = null;
                    return;
                }

                setSession(sess);
                setIsModelLoaded(true);
                document.documentElement.dataset.modelStatus = 'ready';
                document.documentElement.dataset.inferenceStatus = 'startup-ready';
                console.info('Handwriting model startup inference passed', {
                    runtimeVersion: detectedRuntimeVersion,
                    modelUrl: selectedModel.url,
                    modelName: selectedModel.name
                });
                setStatus({ text: t.modelReady, type: "info" });
            } catch (e) {
                if (cancelled) return;
                const message = getErrorMessage(e);
                setLoadingStage(t.loadingFailed);
                document.documentElement.dataset.modelStatus = 'error';
                console.error('Handwriting model startup failed', {
                    runtimeVersion: ort.env?.versions?.web || ONNX_RUNTIME_VERSION,
                    wasmBaseUrl: ONNX_WASM_BASE_URL,
                    modelUrl: selectedModel.url,
                    modelName: selectedModel.name,
                    error: e
                });
                setStatus({ text: t.modelLoadError(message), type: "error" });
            }
        }
        initModel();

        return () => {
            cancelled = true;
            if (loadedSession && typeof loadedSession.release === 'function') {
                loadedSession.release();
                loadedSession = null;
            }
        };
    }, [selectedModelId]);

    const generateWord = useCallback(async (customText = text) => {
        if (!session) return;
        const cleanText = customText.replace(/[^א-ת.,\-!?/)( :;'"'\n]/g, '');
        if (!cleanText) {
            setStatus({ text: t.validCharacters, type: "error" });
            return;
        }

        setStatus({ text: t.generating, type: "info" });
        await new Promise(r => setTimeout(r, 50)); // Ensure browser paints the banner

        const items = [];
        let inferenceFailure = null;
        document.documentElement.dataset.inferenceStatus = 'running';
        for (let i = 0; i < cleanText.length; i++) {
            if (i % 5 === 0) await new Promise(r => setTimeout(r, 0)); // yield to UI

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
            let bestCropInfo = null, bestRawPixels = null;
            while (!validLetter && attempts < 15) {
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

                    if (isValidLetter(char, cropInfo, isQualityCheckEnabled)) {
                        validLetter = true;
                    } else {
                        if (!cropInfo.isEmpty && !bestCropInfo) {
                            bestCropInfo = cropInfo;
                            bestRawPixels = rawPixels;
                        }
                        attempts++;
                    }
                } catch (e) {
                    inferenceFailure = getErrorMessage(e);
                    document.documentElement.dataset.inferenceStatus = 'error';
                    console.error('Inference engine failed while generating text', {
                        runtimeVersion: ort.env?.versions?.web || ONNX_RUNTIME_VERSION,
                        modelUrl: selectedModel.url,
                        modelName: selectedModel.name,
                        character: char,
                        characterIndex: i,
                        error: e
                    });
                    break;
                }
            }

            if (!validLetter && bestCropInfo) {
                cropInfo = bestCropInfo;
                rawPixels = bestRawPixels;
            }

            if (cropInfo && !cropInfo.isEmpty) {
                let nextChar = cleanText[i + 1] || null;
                items.push(computeItemLayout({ type: 'char', char: char, nextChar: nextChar, rawPixels, ...cropInfo }));
            }
        }

        setGeneratedData({
            items,
            baselineY: 64 - getGaussian(16, 2),
            id: Date.now()
        });
        if (inferenceFailure) {
            setStatus({ text: t.generationError(inferenceFailure), type: "error" });
        } else {
            document.documentElement.dataset.inferenceStatus = 'ready';
            setStatus(null);
        }
    }, [session, text, selectedModelId, isQualityCheckEnabled]);

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
        let bestCropInfo = null, bestRawPixels = null;
        while (!validLetter && attempts < 15) {
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

                if (isValidLetter(oldItem.char, cropInfo, isQualityCheckEnabled)) {
                    validLetter = true;
                } else {
                    if (!cropInfo.isEmpty && !bestCropInfo) {
                        bestCropInfo = cropInfo;
                        bestRawPixels = rawPixels;
                    }
                    attempts++;
                }
            } catch (e) {
                const message = getErrorMessage(e);
                document.documentElement.dataset.inferenceStatus = 'error';
                console.error('Inference engine failed while regenerating a character', {
                    runtimeVersion: ort.env?.versions?.web || ONNX_RUNTIME_VERSION,
                    modelUrl: selectedModel.url,
                    modelName: selectedModel.name,
                    character: oldItem.char,
                    characterIndex: index,
                    error: e
                });
                setStatus({ text: t.regenerateError(message), type: "error" });
                break;
            }
        }

        if (!validLetter && bestCropInfo) {
            cropInfo = bestCropInfo;
            rawPixels = bestRawPixels;
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

    // --- Layout Calculation (useMemo) ---
    const layoutData = React.useMemo(() => {
        if (!generatedData) return null;
        const LINE_SPACING = isDebugMode ? 64 : 48;

        const MAX_LINE_WIDTH = 1000;
        let lines = [[]];
        let computedItems = [];
        let maxLineWidth = 0;

        let tokens = [];
        let currentWord = [];

        // Group items into words (characters) and others (spaces/newlines)
        generatedData.items.forEach((item, originalIndex) => {
            let itemWithIdx = { ...item, originalIndex, ref: item };
            if (item.type === 'newline' || item.type === 'space') {
                if (currentWord.length > 0) {
                    tokens.push({ type: 'word', items: currentWord });
                    currentWord = [];
                }
                tokens.push({ type: item.type, items: [itemWithIdx] });
            } else {
                currentWord.push(itemWithIdx);
            }
        });
        if (currentWord.length > 0) {
            tokens.push({ type: 'word', items: currentWord });
        }

        let currentLineWidth = 0;

        tokens.forEach(token => {
            if (token.type === 'newline') {
                lines.push([]);
                computedItems.push({ ...token.items[0], lineIndex: lines.length - 1 });
                currentLineWidth = 0;
            } else if (token.type === 'space') {
                let spaceItem = token.items[0];
                lines[lines.length - 1].push({ ...spaceItem, lineIndex: lines.length - 1 });
                currentLineWidth += spaceItem.finalW;
            } else if (token.type === 'word') {
                let wordWidth = token.items.reduce((sum, item) => sum + item.finalW, 0);

                // Check if we need to wrap
                if (currentLineWidth > 0 && currentLineWidth + wordWidth > MAX_LINE_WIDTH) {
                    lines.push([]);
                    currentLineWidth = 0;
                }

                token.items.forEach(item => {
                    lines[lines.length - 1].push({ ...item, lineIndex: lines.length - 1 });
                });
                currentLineWidth += wordWidth;
            }
        });

        // First pass: just compute canvas width using static bounds so currentX can be set
        lines.forEach(line => {
            let width = 0;
            line.forEach(item => {
                width += item.finalW;
            });
            if (width > maxLineWidth) maxLineWidth = width;
        });

        const canvasWidth = Math.max(10, maxLineWidth + 40);
        const canvasHeight = Math.max(64, lines.length * LINE_SPACING + 20);

        // Second pass: full contour layout
        let baseline = generatedData.baselineY;

        lines.forEach((line, lineIndex) => {
            let currentX = canvasWidth - 20;
            let currentBaseline = baseline + (lineIndex * LINE_SPACING);
            let prevItem = null;
            let prevDrawX = 0;
            let prevItemRange = null;

            line.forEach((item) => {
                let drawX;
                let standardY = item.type !== 'space' ? currentBaseline - (item.yPosTop * LINE_HEIGHT_PX) : currentBaseline;
                let itemRange = item.type !== 'space' ? getItemYRange(item, currentBaseline) : { top: currentBaseline - LINE_HEIGHT_PX, bottom: currentBaseline };

                if (item.type === 'space') {
                    drawX = currentX - item.finalW;
                    currentX = drawX;
                    prevItem = item;
                    prevDrawX = drawX;
                    prevItemRange = itemRange;
                } else {
                    if (!prevItem || prevItem.type === 'space') {
                        drawX = currentX - item.finalW;
                    } else {
                        let maxDrawX = prevDrawX + prevItem.finalW;
                        let targetSpacing = item.spacingJitter || 1.0;

                        let VERTICAL_PAD = 6;
                        let yStart = Math.max(prevItemRange.top, itemRange.top) - VERTICAL_PAD;
                        let yEnd = Math.min(prevItemRange.bottom, itemRange.bottom) + VERTICAL_PAD;

                        let overlapFound = false;
                        if (yStart < yEnd) {
                            let itemRef = item.ref;
                            let prevItemRef = prevItem.ref;
                            let minDist;
                            let itemCache = itemRef && kerningCacheRef.current.get(itemRef);

                            if (itemCache && itemCache.has(prevItemRef)) {
                                minDist = itemCache.get(prevItemRef);
                            } else {
                                minDist = Infinity;
                                for (let y = Math.floor(yStart); y < Math.floor(yEnd); y++) {
                                    let lb = getLeftItemRightBound(item, y, standardY, currentBaseline);
                                    if (lb === item.finalW) continue; // Skip if item has no ink at this exact Y to prevent false spacing

                                    let minRb = Infinity;
                                    for (let dy = -VERTICAL_PAD; dy <= VERTICAL_PAD; dy++) {
                                        let rb = getRightItemLeftBound(prevItem, y + dy, currentBaseline - (prevItem.yPosTop * LINE_HEIGHT_PX), currentBaseline);
                                        if (rb < minRb) minRb = rb;
                                    }
                                    let dist = minRb - lb;
                                    if (dist < minDist) minDist = dist;
                                }
                                if (itemRef && prevItemRef) {
                                    if (!itemCache) {
                                        itemCache = new WeakMap();
                                        kerningCacheRef.current.set(itemRef, itemCache);
                                    }
                                    itemCache.set(prevItemRef, minDist);
                                }
                            }

                            if (minDist !== Infinity) {
                                maxDrawX = prevDrawX + minDist - targetSpacing;
                                overlapFound = true;
                            }
                        }

                        let overlapLimit = prevDrawX;
                        if (item.char === '"') {
                            overlapLimit = prevDrawX - item.finalW / 2;
                        }

                        if (!overlapFound) {
                            maxDrawX = overlapLimit;
                        }

                        if (maxDrawX > overlapLimit) {
                            maxDrawX = overlapLimit;
                        }

                        drawX = maxDrawX;
                    }

                    currentX = drawX;
                    prevItem = item;
                    prevDrawX = drawX;
                    prevItemRange = itemRange;
                }

                computedItems.push({
                    ...item,
                    drawX,
                    standardY,
                    currentBaseline
                });
            });
        });

        return { computedItems, canvasWidth, canvasHeight, linesLength: lines.length, LINE_SPACING };
    }, [generatedData, isDebugMode]);

    // --- The Drawing / Rendering Logic ---
    useEffect(() => {
        if (!layoutData || !canvasRef.current || !wrapperRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        const userColorRGB = hexToRgb(textColor);

        canvas.width = layoutData.canvasWidth;
        canvas.height = layoutData.canvasHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw debug baselines
        if (isDebugMode) {
            let baseline = generatedData.baselineY;
            for (let i = 0; i < layoutData.linesLength; i++) {
                let currentBaseline = baseline + (i * layoutData.LINE_SPACING);
                ctx.strokeStyle = '#38bdf8'; ctx.beginPath(); ctx.moveTo(0, currentBaseline); ctx.lineTo(canvas.width, currentBaseline); ctx.stroke();
                ctx.strokeStyle = '#818cf8'; ctx.beginPath(); ctx.moveTo(0, currentBaseline - LINE_HEIGHT_PX); ctx.lineTo(canvas.width, currentBaseline - LINE_HEIGHT_PX); ctx.stroke();
            }
        }

        const drawScaledLetter = (item, rawWidth, rawHeight, xOffset, yOffset, scale) => {
            let finalW = Math.max(1, Math.floor(rawWidth * scale));
            let finalH = Math.max(1, Math.floor(rawHeight * scale));

            let cacheKey = item.ref || item;
            let cachedCanvas = charCacheRef.current.get(cacheKey);

            if (!cachedCanvas) {
                const SS_FACTOR = 3;
                const highResW = Math.floor(rawWidth * SS_FACTOR);
                const highResH = Math.floor(rawHeight * SS_FACTOR);

                const tinyCanvas = document.createElement('canvas');
                tinyCanvas.width = rawWidth;
                tinyCanvas.height = rawHeight;
                const tinyCtx = tinyCanvas.getContext('2d');
                const tinyImgData = tinyCtx.createImageData(rawWidth, rawHeight);

                let pixels = item.pixels;

                for (let i = 0; i < pixels.length; i++) {
                    let val = pixels[i];
                    if (val < -0.6) val = -1.0; // Transparent background fix
                    let x = (val + 1.0) / 2.0;
                    x = Math.max(0, Math.min(1, x));
                    let grayscale = Math.floor((1.0 - x) * 255);
                    let idx = i * 4;
                    tinyImgData.data[idx] = grayscale;
                    tinyImgData.data[idx + 1] = grayscale;
                    tinyImgData.data[idx + 2] = grayscale;
                    tinyImgData.data[idx + 3] = 255;
                }
                tinyCtx.putImageData(tinyImgData, 0, 0);

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

                cachedCanvas = highResCanvas;
                charCacheRef.current.set(cacheKey, cachedCanvas);
            }

            ctx.globalCompositeOperation = 'source-over';
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(cachedCanvas, xOffset, yOffset, finalW, finalH);
            ctx.globalCompositeOperation = 'source-over';
        };

        const drawHitbox = (x, y, w, h, color) => {
            ctx.fillStyle = color + '33';
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = color + 'CC';
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y, w, h);
        };

        layoutData.computedItems.forEach((item) => {
            if (item.type === 'newline') return;

            let drawX = item.drawX;
            let currentBaseline = item.currentBaseline;

            if (item.type === 'space') {
                if (isDebugMode) drawHitbox(drawX, currentBaseline - LINE_HEIGHT_PX, item.finalW, LINE_HEIGHT_PX, '#10b981');
            }
            else if (item.char === ':') {
                let topY = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (item.finalH / 2);
                let bottomY = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
                drawScaledLetter(item, item.cropW, item.cropH, drawX, topY, item.scaleFactor);
                drawScaledLetter(item, item.cropW, item.cropH, drawX, bottomY, item.scaleFactor);
                if (isDebugMode) { drawHitbox(drawX, topY, item.finalW, item.finalH, '#ef4444'); drawHitbox(drawX, bottomY, item.finalW, item.finalH, '#ef4444'); }
            }
            else if (item.char === ';') {
                let dotScale = item.scaleFactor * 0.6;
                let dotW = item.cropH * dotScale;
                let dotH = item.cropH * dotScale;
                let topY = currentBaseline - (0.8 * LINE_HEIGHT_PX) - (dotH / 2);
                let bottomY = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
                let topCenterX = drawX + (item.finalW - dotW) / 2;
                drawScaledLetter(item, item.cropW, item.cropH, topCenterX, topY, dotScale);
                drawScaledLetter(item, item.cropW, item.cropH, drawX, bottomY, item.scaleFactor);
                if (isDebugMode) { drawHitbox(topCenterX, topY, dotW, dotH, '#ef4444'); drawHitbox(drawX, bottomY, item.finalW, item.finalH, '#ef4444'); }
            }
            else if (item.char === '"') {
                let yPos = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
                let halfW = item.finalW / 2;
                drawScaledLetter(item, item.cropW, item.cropH, drawX + halfW, yPos, item.scaleFactor);
                drawScaledLetter(item, item.cropW, item.cropH, drawX, yPos, item.scaleFactor);
                if (isDebugMode) { drawHitbox(drawX + halfW, yPos, halfW, item.finalH, '#ef4444'); drawHitbox(drawX, yPos, halfW, item.finalH, '#ef4444'); }
            }
            else {
                let yPos = currentBaseline - (item.yPosTop * LINE_HEIGHT_PX);
                drawScaledLetter(item, item.cropW, item.cropH, drawX, yPos, item.scaleFactor);
                if (isDebugMode) drawHitbox(drawX, yPos, item.finalW, item.finalH, '#ef4444');
            }
        });

    }, [layoutData, isDebugMode, isCurveEnabled, isPaperBlack, textColor]);

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
            setStatus({ text: t.quoteLoadError, type: "error" });
        }
    };

    const processImage = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setStatus({ text: t.ocrProcessing, type: "info" });
        try {
            const tesseract = await loadTesseract();
            const result = await tesseract.recognize(file, 'heb');
            const extractedText = result.data.text.trim();
            setText(extractedText);
            generateWord(extractedText);
        } catch (err) {
            setStatus({ text: t.ocrError, type: "error" });
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
                setStatus({ text: t.copied, type: "success" });
                setTimeout(() => setStatus(null), 2500);
            } catch (e) {
                setStatus({ text: t.copyError, type: "error" });
            }
        });
    };

    return (
        <div className="app-container">
            <div className={`loading-overlay ${isModelLoaded ? 'fade-out' : ''}`}>
                <div className="spinner"></div>
                <div className="loading-text">{loadingStage}</div>
            </div>

            <div className="bg-orb orb-1"></div>
            <div className="bg-orb orb-2"></div>

            <header className="glass-header">
                <div className="header-content">
                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '15px' }}>
                        <h1 style={{ margin: 0 }}>{t.header} <span>{t.accent}</span></h1>
                        <button className="tool-btn theme-toggle" onClick={toggleTheme} title={t.changeTheme} style={{ border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                            <img src={isDarkMode ? "assets/sun.png" : "assets/moon.png"} alt={t.themeIcon} className="icon-img" />
                        </button>
                    </div>
                    <p>{t.subtitle}</p>
                </div>
            </header>

            <main className="main-content">
                <section className="glass-card controls-card">
                    <h2>{t.prompt}</h2>
                    <div className="input-row">
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            onKeyDown={handleKeyPress}
                            dir="auto"
                            placeholder={t.inputPlaceholder}
                            className="text-input"
                            style={{ resize: 'vertical', minHeight: '60px', flex: 1, fontFamily: 'inherit', fontSize: '1.1rem', padding: '10px' }}
                        />
                        <button className="btn-primary" onClick={() => generateWord()} title={t.write}>
                            <img src="assets/write.png" alt={t.writeIcon} className="icon-img" />
                        </button>
                    </div>

                    <div className="toolbar">
                        <label className="tool-btn" title={t.uploadOcr}>
                            <input type="file" accept="image/*" style={{ display: 'none' }} onChange={processImage} />
                            <img src="assets/picture.png" alt={t.ocrIcon} className="icon-img" />
                        </label>

                        <button className="tool-btn" onClick={loadRandomQuote} title={t.randomQuote}>
                            <img src="assets/quote-right.png" alt={t.quoteIcon} className="icon-img" />
                        </button>

                        <label className="model-select-wrapper tool-btn" title={`${t.selectModel} — ${selectedModel.name}`}>
                            <img src="assets/model-select.png" alt={t.selectModel} className="icon-img" />
                            <select
                                value={selectedModelId}
                                onChange={(event) => setSelectedModelId(event.target.value)}
                                aria-label={t.selectModel}
                            >
                                {MODELS.map(model => (
                                    <option key={model.id} value={model.id}>{model.name}</option>
                                ))}
                            </select>
                        </label>

                        <button
                            className="tool-btn language-toggle"
                            onClick={() => setLanguage(current => current === 'he' ? 'en' : 'he')}
                            title={t.languageLabel}
                            aria-label={t.languageLabel}
                        >
                            <span>{t.language}</span>
                        </button>

                        <div className="divider"></div>

                        <button className="tool-btn" onClick={copyToClipboard} title={t.copy}>
                            <img src="assets/copy.png" alt={t.copyIcon} className="icon-img" />
                        </button>

                        <div className="divider"></div>

                        <button className={`tool-btn ${isDebugMode ? 'active' : ''}`} onClick={() => setIsDebugMode(!isDebugMode)} title={t.debug}>
                            <img src="assets/debug.png" alt={t.debugIcon} className="icon-img" />
                        </button>

                        <button className={`tool-btn ${isQualityCheckEnabled ? 'active' : ''}`} onClick={() => setIsQualityCheckEnabled(!isQualityCheckEnabled)} title={t.quality}>
                            <img src="assets/check-circle.png" alt={t.qualityIcon} className="icon-img" />
                        </button>

                        <button className={`tool-btn ${isPaperBlack ? 'active' : ''}`} onClick={togglePaperColor} title={t.paper}>
                            <img src="assets/invert.png" alt={t.paperIcon} className="icon-img" />
                        </button>

                        <div className="color-picker-wrapper tool-btn" title={t.inkColor}>
                            <input
                                type="color"
                                value={draftTextColor}
                                onChange={handleColorChange}
                                className="color-wheel"
                            />
                        </div>

                        <button className="tool-btn" onClick={() => generateWord()} title={t.refresh}>
                            <img src="assets/rotate-right.png" alt={t.refreshIcon} className="icon-img" />
                        </button>
                    </div>

                    {status && (
                        <div className={`status-pill ${status.type}`}>
                            {status.type === 'error' && <img src="assets/error.png" alt={t.errorIcon} width="14" height="14" style={{ filter: 'var(--icon-filter)' }} />}
                            {status.type === 'info' && <img src="assets/exclamation.png" alt={t.infoIcon} width="14" height="14" style={{ filter: 'var(--icon-filter)' }} />}
                            <span>{status.text}</span>
                        </div>
                    )}
                </section>

                <section className="glass-card canvas-card">
                    <h2>{t.result}</h2>
                    <div className="canvas-overflow-wrapper">
                        <div className="canvas-inner-wrapper" ref={wrapperRef}>
                            <canvas ref={canvasRef} className={`main-canvas ${isPaperBlack ? 'paper-black' : 'paper-white'}`} />
                            {isDebugMode && layoutData && (
                                <div className="debug-buttons-layer">
                                    {layoutData.computedItems.map((item) => {
                                        if (item.type === 'newline') return null;

                                        // Position the button exactly over the drawn character width
                                        // Wait, layoutData is drawn to canvas, but the canvas CSS width might be different?
                                        // No, the canvas wrapper handles scaling, so absolute pixel coordinates matching canvas size work perfectly here 
                                        // since the wrapper and canvas size match in the CSS.
                                        let myRight = layoutData.canvasWidth - item.drawX - item.finalW;
                                        let currTop = item.currentBaseline - 55;

                                        return (
                                            <button
                                                key={`regen-${item.originalIndex}-${item.id || item.originalIndex}`}
                                                className="regen-char-btn"
                                                style={{
                                                    right: `${myRight}px`,
                                                    top: `${currTop}px`,
                                                    width: `${item.finalW}px`
                                                }}
                                                onClick={() => regenerateSingleChar(item.originalIndex)}
                                                    title={t.regenerate}
                                            >
                                                <img src="assets/rotate-right.png" alt={t.regenerate} style={{ width: '10px', height: '10px', filter: 'var(--icon-filter)' }} />
                                            </button>
                                        );
                                    })}
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
