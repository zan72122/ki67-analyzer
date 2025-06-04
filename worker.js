let cv;

self.importScripts('https://docs.opencv.org/4.5.5/opencv.js');

self.onmessage = async function(e) {
    const { type, data } = e.data;
    
    if (type === 'init') {
        await waitForCV();
        self.postMessage({ type: 'ready' });
    } else if (type === 'process') {
        const result = await processImageChunk(data);
        self.postMessage({ type: 'result', data: result });
    }
};

async function waitForCV() {
    return new Promise((resolve) => {
        if (cv && cv.Mat) {
            resolve();
        } else {
            cv = self.cv;
            if (cv && cv.Mat) {
                resolve();
            } else {
                setTimeout(() => waitForCV().then(resolve), 100);
            }
        }
    });
}

async function processImageChunk(data) {
    const { imageData, threshold, minSize, maxSize, chunkInfo } = data;
    
    const mat = cv.matFromImageData(imageData);
    const gray = new cv.Mat();
    const binary = new cv.Mat();
    
    cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
    cv.threshold(gray, binary, threshold, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    
    const nuclei = [];
    
    for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        
        if (area >= minSize && area <= maxSize) {
            const moments = cv.moments(contour);
            const cx = moments.m10 / moments.m00;
            const cy = moments.m01 / moments.m00;
            const rect = cv.boundingRect(contour);
            
            const roiGray = gray.roi(rect);
            const meanIntensity = cv.mean(roiGray)[0];
            
            nuclei.push({
                id: Date.now() + Math.random(),
                x: cx + chunkInfo.offsetX,
                y: cy + chunkInfo.offsetY,
                width: rect.width,
                height: rect.height,
                area: area,
                intensity: meanIntensity,
                isPositive: meanIntensity < threshold * 0.7
            });
            
            roiGray.delete();
        }
        contour.delete();
    }
    
    mat.delete();
    gray.delete();
    binary.delete();
    contours.delete();
    hierarchy.delete();
    
    return nuclei;
}