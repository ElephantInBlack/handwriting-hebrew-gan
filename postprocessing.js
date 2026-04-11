class ImageProcessor {
    constructor(a = 0.9, b = 2) {
        this.a = a;
        this.b = b;
    }

    // Mathematical Contrast Curve from Desmos
    applyContrastCurve(val_tanh) { 
        // 1. Transform [-1, 1] range to [0, 1]
        
        let x = (val_tanh + 1.0) / 2.0;

        x = Math.max(0, Math.min(1, x)); // Clamp to ensure valid range
        
        // 2. Apply formula: max(-h(1-x) + 1, 0)
        let v = 1.0 - x; // This is (1-x)

        // Calculate g(v) = (2v^3 - 3av^2) / (2 - 3a)
        let num = 2.0 * Math.pow(v, 3) - 3.0 * this.a * Math.pow(v, 2);
        let den = 2.0 - 3.0 * this.a;
        let g_v = num / den;
        
        // Ensure non-negative before applying power
        g_v = Math.max(0, g_v); 

        // Calculate h(1-x) = (g(v))^b
        let h_v = Math.pow(g_v, this.b);

        // Final output value: max(-h + 1, 0)
        let y = Math.max(0, 1.0 - h_v);

        // 3. Invert colors (0 = Black Ink, 1 = White Background) and scale to 255
        let pixelValue = Math.floor((1.0 - y) * 255);
        return Math.max(0, Math.min(255, pixelValue)); 
    }

    // Processes the raw tensor, applies math, and crops white space
    // Processes the raw tensor, applies math, and crops white space
    processAndCrop(flatPixelArray, cropThreshold = 250) {
        const width = 64;
        const height = 64;
        let minX = width, maxX = 0, minY = height, maxY = 0;
        let processedPixels = new Uint8Array(width * height);
        
        let inkPixelCount = 0; // NEW: Track the exact amount of ink

        // Pass 1: Apply color correction and find bounding box of the "ink"
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let idx = y * width + x;
                let val = this.applyContrastCurve(flatPixelArray[idx]);
                processedPixels[idx] = val;

                // If pixel is darker than threshold, it's ink! Update crop bounds.
                if (val < cropThreshold) {
                    inkPixelCount++; // Count this ink pixel
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        // --- NEW QUALITY CONTROL (Density & Ink based) ---
        let boxW = maxX - minX + 1;
        let boxH = maxY - minY + 1;
        let boxArea = boxW * boxH;
        let density = inkPixelCount / boxArea;

        // Failsafe conditions:
        let isEmpty = (minX > maxX || minY > maxY);
        // "Ghost": Huge bounding box but mostly empty (catches faint edge artifacts)
        let isGhost = (!isEmpty && boxArea > 100 && density < 0.05); 
        // "Smudge": Just a tiny speck of noise (less than 8 pixels total ink)
        let isSmudge = (!isEmpty && inkPixelCount < 8);

        if (isEmpty || isGhost || isSmudge) {
            return { isEmpty: true, pixels: processedPixels, cropW: width, cropH: height, origMinY: 0 };
        }

        // Add 2px padding to the crop box to preserve smooth edges
        minX = Math.max(0, minX - 2);
        maxX = Math.min(width - 1, maxX + 2);
        minY = Math.max(0, minY - 2);
        maxY = Math.min(height - 1, maxY + 2);

        let cropW = maxX - minX + 1;
        let cropH = maxY - minY + 1;

        // Pass 2: Extract just the cropped area
        let croppedPixels = new Uint8Array(cropW * cropH);
        for (let cy = 0; cy < cropH; cy++) {
            for (let cx = 0; cx < cropW; cx++) {
                let origY = minY + cy;
                let origX = minX + cx;
                croppedPixels[cy * cropW + cx] = processedPixels[origY * width + origX];
            }
        }

        // Return successfully! 
        return { isEmpty: false, pixels: croppedPixels, cropW, cropH, origMinY: minY };
    }
}