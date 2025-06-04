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
    const gray = extractDAB(mat);
    cv.medianBlur(gray, gray, 3);
    const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
    clahe.apply(gray, gray);

    const binary = new cv.Mat();
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
            
            const mask = new cv.Mat.zeros(gray.rows, gray.cols, cv.CV_8UC1);
            const cVec = new cv.MatVector();
            cVec.push_back(contour);
            cv.drawContours(mask, cVec, 0, new cv.Scalar(255), -1);
            const meanIntensity = cv.mean(gray, mask)[0];
            
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
            
            cVec.delete();
            mask.delete();
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

function extractDAB(srcRGBMat) {
    let od = new cv.Mat();
    srcRGBMat.convertTo(od, cv.CV_32F, 1/255.0);
    cv.add(od, new cv.Scalar(1.0, 1.0, 1.0), od);
    cv.log(od, od);
    cv.multiply(od, new cv.Scalar(-1.0, -1.0, -1.0), od);

    const dab = new cv.Mat();
    const kernel = cv.matFromArray(1, 3, cv.CV_32F, [0.650, 0.704, 0.286]);
    cv.transform(od, dab, kernel);

    let dab8 = new cv.Mat();
    cv.normalize(dab, dab8, 0, 255, cv.NORM_MINMAX);
    dab8.convertTo(dab8, cv.CV_8U);

    od.delete();
    dab.delete();
    kernel.delete();

    return dab8;
}