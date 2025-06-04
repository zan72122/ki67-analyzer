class Ki67Analyzer {
    constructor() {
        this.image = null;
        this.nuclei = [];
        this.history = [];
        this.historyIndex = -1;
        this.idleTimer = null;
        this.cvReady = false;
        this.workers = [];
        this.workerCount = navigator.hardwareConcurrency || 4;
        
        this.initializeElements();
        this.bindEvents();
        this.loadSettings();
        this.waitForOpenCV();
        this.initializeWorkers();
    }
    
    initializeElements() {
        this.dropzone = document.getElementById('dropzone');
        this.fileInput = document.getElementById('fileInput');
        this.browseBtn = document.getElementById('browseBtn');
        this.workspace = document.getElementById('workspace');
        this.imageCanvas = document.getElementById('imageCanvas');
        this.overlayCanvas = document.getElementById('overlayCanvas');
        this.imageCtx = this.imageCanvas.getContext('2d');
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        
        this.undoBtn = document.getElementById('undo');
        this.redoBtn = document.getElementById('redo');
        this.resetBtn = document.getElementById('reset');
        this.downloadBtn = document.getElementById('downloadCSV');
        this.loadDemoBtn = document.getElementById('loadDemo');
        
        this.positiveCount = document.getElementById('positiveCount');
        this.negativeCount = document.getElementById('negativeCount');
        this.ki67Index = document.getElementById('ki67Index');
        
        this.minSizeSlider = document.getElementById('minSize');
        this.maxSizeSlider = document.getElementById('maxSize');
        this.thresholdSlider = document.getElementById('threshold');
        this.minSizeValue = document.getElementById('minSizeValue');
        this.maxSizeValue = document.getElementById('maxSizeValue');
        this.thresholdValue = document.getElementById('thresholdValue');
    }
    
    bindEvents() {
        this.dropzone.addEventListener('dragover', this.handleDragOver.bind(this));
        this.dropzone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.dropzone.addEventListener('drop', this.handleDrop.bind(this));
        this.browseBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        this.overlayCanvas.addEventListener('click', this.handleCanvasClick.bind(this));
        
        this.undoBtn.addEventListener('click', this.undo.bind(this));
        this.redoBtn.addEventListener('click', this.redo.bind(this));
        this.resetBtn.addEventListener('click', this.reset.bind(this));
        this.downloadBtn.addEventListener('click', this.downloadCSV.bind(this));
        this.loadDemoBtn.addEventListener('click', this.loadDemoImage.bind(this));
        
        this.minSizeSlider.addEventListener('input', this.updateParameter.bind(this));
        this.maxSizeSlider.addEventListener('input', this.updateParameter.bind(this));
        this.thresholdSlider.addEventListener('input', this.updateParameter.bind(this));
        
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
                e.preventDefault();
                if (e.shiftKey) {
                    this.redo();
                } else {
                    this.undo();
                }
            }
        });
    }
    
    waitForOpenCV() {
        if (typeof cv !== 'undefined') {
            this.cvReady = true;
            console.log('OpenCV.js ready');
        } else {
            setTimeout(() => this.waitForOpenCV(), 100);
        }
    }
    
    async initializeWorkers() {
        for (let i = 0; i < this.workerCount; i++) {
            const worker = new Worker('worker.js');
            await new Promise((resolve) => {
                worker.onmessage = (e) => {
                    if (e.data.type === 'ready') {
                        resolve();
                    }
                };
                worker.postMessage({ type: 'init' });
            });
            this.workers.push(worker);
        }
        console.log(`Initialized ${this.workerCount} workers`);
    }
    
    handleDragOver(e) {
        e.preventDefault();
        this.dropzone.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        this.dropzone.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.dropzone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            this.loadImage(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            this.loadImage(file);
        }
    }
    
    async loadImage(file) {
        const reader = new FileReader();
        reader.onload = async (e) => {
            const img = new Image();
            img.onload = () => {
                this.image = img;
                this.displayImage();
                this.processImage();
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    displayImage() {
        this.imageCanvas.width = this.image.width;
        this.imageCanvas.height = this.image.height;
        this.overlayCanvas.width = this.image.width;
        this.overlayCanvas.height = this.image.height;
        
        this.imageCtx.drawImage(this.image, 0, 0);
        
        this.dropzone.style.display = 'none';
        this.workspace.classList.remove('hidden');
    }
    
    async processImage() {
        if (!this.cvReady && this.workers.length === 0) {
            console.log('Waiting for OpenCV...');
            setTimeout(() => this.processImage(), 100);
            return;
        }
        
        if (this.workers.length > 0) {
            await this.processImageWithWorkers();
        } else {
            await this.processImageLocal();
        }
        
        this.saveState();
        this.render();
        this.updateStats();
    }
    
    async processImageLocal() {
        const src = cv.imread(this.imageCanvas);
        const gray = new cv.Mat();
        const binary = new cv.Mat();
        
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        
        const threshold = parseInt(this.thresholdSlider.value);
        cv.threshold(gray, binary, threshold, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
        
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        
        this.nuclei = [];
        const minSize = parseInt(this.minSizeSlider.value);
        const maxSize = parseInt(this.maxSizeSlider.value);
        
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
                
                this.nuclei.push({
                    id: Date.now() + Math.random(),
                    x: cx,
                    y: cy,
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
        
        src.delete();
        gray.delete();
        binary.delete();
        contours.delete();
        hierarchy.delete();
    }
    
    async processImageWithWorkers() {
        const chunkHeight = Math.ceil(this.image.height / this.workerCount);
        const promises = [];
        
        for (let i = 0; i < this.workerCount; i++) {
            const y = i * chunkHeight;
            const height = Math.min(chunkHeight, this.image.height - y);
            
            if (height <= 0) continue;
            
            const canvas = document.createElement('canvas');
            canvas.width = this.image.width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(this.image, 0, y, this.image.width, height, 0, 0, this.image.width, height);
            
            const imageData = ctx.getImageData(0, 0, this.image.width, height);
            
            const promise = new Promise((resolve) => {
                const worker = this.workers[i];
                worker.onmessage = (e) => {
                    if (e.data.type === 'result') {
                        resolve(e.data.data);
                    }
                };
                
                worker.postMessage({
                    type: 'process',
                    data: {
                        imageData,
                        threshold: parseInt(this.thresholdSlider.value),
                        minSize: parseInt(this.minSizeSlider.value),
                        maxSize: parseInt(this.maxSizeSlider.value),
                        chunkInfo: { offsetX: 0, offsetY: y }
                    }
                });
            });
            
            promises.push(promise);
        }
        
        const results = await Promise.all(promises);
        this.nuclei = results.flat();
    }
    
    handleCanvasClick(e) {
        const rect = this.overlayCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const scale = this.overlayCanvas.width / rect.width;
        const canvasX = x * scale;
        const canvasY = y * scale;
        
        const clickRadius = 20;
        let nucleusFound = false;
        
        for (const nucleus of this.nuclei) {
            const distance = Math.sqrt(
                Math.pow(canvasX - nucleus.x, 2) + 
                Math.pow(canvasY - nucleus.y, 2)
            );
            
            if (distance < Math.max(nucleus.width, nucleus.height) / 2 + clickRadius) {
                nucleus.isPositive = !nucleus.isPositive;
                nucleusFound = true;
                break;
            }
        }
        
        if (nucleusFound) {
            this.saveState();
            this.render();
            this.updateStats();
            this.scheduleAutoDownload();
        }
    }
    
    render() {
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        
        for (const nucleus of this.nuclei) {
            this.overlayCtx.strokeStyle = nucleus.isPositive ? '#e74c3c' : '#2ecc71';
            this.overlayCtx.fillStyle = nucleus.isPositive ? 
                'rgba(231, 76, 60, 0.2)' : 'rgba(46, 204, 113, 0.2)';
            this.overlayCtx.lineWidth = 2;
            
            this.overlayCtx.beginPath();
            this.overlayCtx.ellipse(
                nucleus.x, 
                nucleus.y, 
                nucleus.width / 2, 
                nucleus.height / 2, 
                0, 0, Math.PI * 2
            );
            this.overlayCtx.stroke();
            this.overlayCtx.fill();
        }
    }
    
    updateStats() {
        const positive = this.nuclei.filter(n => n.isPositive).length;
        const negative = this.nuclei.filter(n => !n.isPositive).length;
        const total = this.nuclei.length;
        const index = total > 0 ? (positive / total * 100).toFixed(1) : 0;
        
        this.positiveCount.textContent = positive;
        this.negativeCount.textContent = negative;
        this.ki67Index.textContent = `${index}%`;
    }
    
    updateParameter(e) {
        const slider = e.target;
        const valueSpan = document.getElementById(slider.id + 'Value');
        valueSpan.textContent = slider.value;
        
        this.saveSettings();
        this.processImage();
    }
    
    saveState() {
        const state = JSON.stringify(this.nuclei);
        this.history = this.history.slice(0, this.historyIndex + 1);
        this.history.push(state);
        this.historyIndex++;
        
        this.undoBtn.disabled = this.historyIndex <= 0;
        this.redoBtn.disabled = this.historyIndex >= this.history.length - 1;
    }
    
    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.nuclei = JSON.parse(this.history[this.historyIndex]);
            this.render();
            this.updateStats();
            this.updateHistoryButtons();
        }
    }
    
    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.nuclei = JSON.parse(this.history[this.historyIndex]);
            this.render();
            this.updateStats();
            this.updateHistoryButtons();
        }
    }
    
    updateHistoryButtons() {
        this.undoBtn.disabled = this.historyIndex <= 0;
        this.redoBtn.disabled = this.historyIndex >= this.history.length - 1;
    }
    
    reset() {
        this.processImage();
    }
    
    scheduleAutoDownload() {
        clearTimeout(this.idleTimer);
        this.idleTimer = setTimeout(() => {
            this.downloadCSV();
        }, 500);
    }
    
    downloadCSV() {
        const csv = this.generateCSV();
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ki67_analysis_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    generateCSV() {
        let csv = 'Nucleus ID,X Position,Y Position,Width,Height,Area,Intensity,Classification\n';
        
        this.nuclei.forEach((nucleus, index) => {
            csv += `${index + 1},${nucleus.x.toFixed(2)},${nucleus.y.toFixed(2)},`;
            csv += `${nucleus.width},${nucleus.height},${nucleus.area.toFixed(2)},`;
            csv += `${nucleus.intensity.toFixed(2)},${nucleus.isPositive ? 'Positive' : 'Negative'}\n`;
        });
        
        csv += '\nSummary\n';
        const positive = this.nuclei.filter(n => n.isPositive).length;
        const negative = this.nuclei.filter(n => !n.isPositive).length;
        const total = this.nuclei.length;
        const index = total > 0 ? (positive / total * 100).toFixed(1) : 0;
        
        csv += `Total Nuclei,${total}\n`;
        csv += `Positive (Ki-67+),${positive}\n`;
        csv += `Negative (Ki-67-),${negative}\n`;
        csv += `Ki-67 Index,${index}%\n`;
        
        return csv;
    }
    
    async saveSettings() {
        const settings = {
            minSize: this.minSizeSlider.value,
            maxSize: this.maxSizeSlider.value,
            threshold: this.thresholdSlider.value
        };
        
        localStorage.setItem('ki67_settings', JSON.stringify(settings));
        
        if ('indexedDB' in window) {
            try {
                const db = await this.openDB();
                const transaction = db.transaction(['settings'], 'readwrite');
                const store = transaction.objectStore('settings');
                await store.put({ id: 'parameters', ...settings });
            } catch (err) {
                console.log('IndexedDB save failed, using localStorage only');
            }
        }
    }
    
    async loadSettings() {
        let settings = null;
        
        if ('indexedDB' in window) {
            try {
                const db = await this.openDB();
                const transaction = db.transaction(['settings'], 'readonly');
                const store = transaction.objectStore('settings');
                const result = await store.get('parameters');
                if (result) settings = result;
            } catch (err) {
                console.log('IndexedDB load failed, falling back to localStorage');
            }
        }
        
        if (!settings) {
            const saved = localStorage.getItem('ki67_settings');
            if (saved) settings = JSON.parse(saved);
        }
        
        if (settings) {
            this.minSizeSlider.value = settings.minSize;
            this.maxSizeSlider.value = settings.maxSize;
            this.thresholdSlider.value = settings.threshold;
            
            this.minSizeValue.textContent = settings.minSize;
            this.maxSizeValue.textContent = settings.maxSize;
            this.thresholdValue.textContent = settings.threshold;
        }
    }
    
    openDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('Ki67AnalyzerDB', 1);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('settings')) {
                    db.createObjectStore('settings', { keyPath: 'id' });
                }
            };
        });
    }
    
    async loadDemoImage() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 800;
        canvas.height = 600;
        
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const nucleiData = [];
        for (let i = 0; i < 100; i++) {
            const x = Math.random() * (canvas.width - 50) + 25;
            const y = Math.random() * (canvas.height - 50) + 25;
            const size = Math.random() * 20 + 15;
            const isPositive = Math.random() > 0.5;
            const intensity = isPositive ? Math.random() * 50 + 50 : Math.random() * 50 + 150;
            
            nucleiData.push({ x, y, size, isPositive, intensity });
            
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, size);
            const color = Math.floor(255 - intensity);
            gradient.addColorStop(0, `rgb(${color}, ${color}, ${color})`);
            gradient.addColorStop(1, `rgb(${color + 30}, ${color + 30}, ${color + 30})`);
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
        }
        
        canvas.toBlob((blob) => {
            const file = new File([blob], 'demo-ki67.png', { type: 'image/png' });
            this.loadImage(file);
        });
    }
}

new Ki67Analyzer();