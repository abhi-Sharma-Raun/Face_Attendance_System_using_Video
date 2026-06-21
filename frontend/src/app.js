import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';

// --- Configuration & API Endpoints ---
const BACKEND_URL = import.meta.env.VITE_BACKEND_BASE_URL;
const REGISTRATION_ENDPOINT = `${BACKEND_URL}/${import.meta.env.VITE_REGISTRATION_ENDPOINT}`;
const ATTENDANCE_ENDPOINT = `${BACKEND_URL}/${import.meta.env.VITE_ATTENDANCE_ENDPOINT}`;

// Global State Matrix Object
let state = {
    currentView: "home",
    scanComplete: false,
    activeSession: {
        selectedFrames: {} // Holds blob objects for 'front', 'left', 'right'
    },
    attendanceFile: null
};

// MediaPipe Object Declarations
let faceLandmarker = null;
let webcamStream = null;
let animationFrameId = null;

// Initialization Hook
window.addEventListener('DOMContentLoaded', async () => {
    await initializeFaceLandmarker();
});

/**
 * Initializes the MediaPipe Wasm Face Landmarker Model Task Instance
 */
async function initializeFaceLandmarker() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "CPU"
            },
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: true,
            runningMode: "VIDEO",
            numFaces: 1
        });
        console.log("MediaPipe FaceLandmarker successfully initialized.");
    } catch (error) {
        console.error("Failed to load FaceLandmarker WASM context:", error);
        alert("Critical failure initializing client inference runtime.");
    }
}

/**
 * Handle Application View Transitions (Exposed explicitly to the global window)
 */
export function navigateTo(viewName) {
    state.currentView = viewName;
    document.querySelectorAll('.view-section').forEach(el => el.classList.add('view-hidden'));
    
    const TargetView = document.getElementById(`view-${viewName}`);
    if (TargetView) TargetView.classList.remove('view-hidden');
}

export function backToHome() {
    stopWebcamStream();
    resetScanState();
    navigateTo('home');
}

// Bind navigation systems to window scope so inline HTML onclick handlers can access them
window.navigateTo = navigateTo;
window.backToHome = backToHome;

/**
 * Registration Sub-Module: Core Camera Capture & Pose Tracking Loops
 */
async function startVerificationScan() {
    const name = document.getElementById('reg-name').value.trim();
    const email = document.getElementById('reg-email').value.trim();

    if (!name || !email) {
        alert("Please enter your name and email before turning on the webcam scan.");
        return;
    }

    if (!faceLandmarker) {
        alert("Inference engine asset is loading. Please give it a few seconds...");
        return;
    }

    state.activeSession = {
        selectedFrames: {}
    };
    state.scanComplete = false;

    document.getElementById('capture-hud').innerText = "Captured: []";

    document.getElementById('btn-start-scan').classList.add('view-hidden');
    document.getElementById('scan-active-status').classList.remove('view-hidden');
    document.getElementById('scan-success-status').classList.add('view-hidden');
    document.getElementById('video-wrapper').classList.remove('view-hidden');

    const video = document.getElementById('webcam');
    
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
            audio: false
        });
        video.srcObject = webcamStream;
        video.addEventListener('loadeddata', () => {
            video.play();
            animationFrameId = requestAnimationFrame(renderTrackingLoop);
        });
    } catch (err) {
        console.error("Camera resolution setup error: ", err);
        alert("Could not access the webcam. Please verify device sandbox permissions.");
        resetScanState();
    }
}
window.startVerificationScan = startVerificationScan;

/**
 * Continuous WebRTC Render Frame Parsing Analytics Execution
 */
async function renderTrackingLoop() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');

    if (!video.videoWidth || video.paused || video.ended) {
        console.log("video frame/width is not proper")
        animationFrameId = requestAnimationFrame(renderTrackingLoop);
        return;
    }

    if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let nowInMs = Date.now();

    const result = faceLandmarker.detectForVideo(video, nowInMs);

    try{
        if (result && result.facialTransformationMatrixes[0].data){
            console.log("fine")
        }
    }
    catch{
        console.log(result)
    }
    if (result && result.facialTransformationMatrixes && result.facialTransformationMatrixes.length>0) {
        
        const matrixObj = result.facialTransformationMatrixes[0];
        
        if (matrixObj && matrixObj.data) {
            const matrixData = matrixObj.data;

            const r20 = matrixData[8];
            const r21 = matrixData[9];
            const r22 = matrixData[10];

            const yawRadians = Math.atan2(-r20, Math.sqrt(Math.pow(r21, 2) + Math.pow(r22, 2)));
            const yawDegrees = yawRadians * (180 / Math.PI);

            let detectedPose = null;
            const selectedPoses = Object.keys(state.activeSession.selectedFrames);

            if (yawDegrees >= -10 && yawDegrees <= 10 && !selectedPoses.includes("front")) {
                detectedPose = "front";
            } else if (yawDegrees > 30 && !selectedPoses.includes("right")) {
                detectedPose = "right";
            } else if (yawDegrees < -30 && !selectedPoses.includes("left")) {
                detectedPose = "left";
            }

            if (detectedPose && !selectedPoses.includes(detectedPose)) {
                console.log(`${detectedPose.toUpperCase()} PROFILE DETECTED - Processing capture...`);
                
                const frameBlob = await captureCanvasFrameBlob(video);
                state.activeSession.selectedFrames[detectedPose] = frameBlob;
                
                document.getElementById('capture-hud').innerText = `Captured: [${Object.keys(state.activeSession.selectedFrames).join(', ')}]`;
            }
        }
    }
    else{
        console.log("Some problem here")
    }

    if (Object.keys(state.activeSession.selectedFrames).length === 3) {
        console.log("ALL profiles captured")
        state.scanComplete = true;
        stopWebcamStream();
        
        document.getElementById('video-wrapper').classList.add('view-hidden');
        document.getElementById('scan-active-status').classList.add('view-hidden');
        document.getElementById('btn-start-scan').classList.add('view-hidden');
        document.getElementById('scan-success-status').classList.remove('view-hidden');
        return;
    }

    animationFrameId = requestAnimationFrame(renderTrackingLoop);
}

/**
 * Capture frame directly from raw video stream pipeline converting into standard JPEG image blob
 */
function captureCanvasFrameBlob(videoElement) {
    return new Promise((resolve) => {
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = videoElement.videoWidth;
        captureCanvas.height = videoElement.videoHeight;
        const ctx = captureCanvas.getContext('2d');
        
        ctx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
        captureCanvas.toBlob((blob) => {
            resolve(blob);
        }, 'image/jpeg', 0.95);
    });
}

function stopWebcamStream() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
}

function resetScanState() {
    stopWebcamStream();
    state.scanComplete = false;
    state.activeSession.selectedFrames = {};
    
    document.getElementById('capture-hud').innerText = "Captured: []";
    document.getElementById('video-wrapper').classList.add('view-hidden');
    document.getElementById('scan-active-status').classList.add('view-hidden');
    document.getElementById('scan-success-status').classList.add('view-hidden');
    document.getElementById('btn-start-scan').classList.remove('view-hidden');
}
window.resetScanState = resetScanState;

/**
 * Multi-Part API Form Pipeline Submission handler parsing student payloads
 */
async function submitRegistration() {
    const name = document.getElementById('reg-name').value.trim();
    const email = document.getElementById('reg-email').value.trim();
    const submitBtn = document.getElementById('btn-submit-registration');

    if (!name || !email) {
        alert("Please provide both name and email.");
        return;
    }
    if (!state.scanComplete || Object.keys(state.activeSession.selectedFrames).length !== 3) {
        alert("Please complete the automated verification scan before submitting.");
        return;
    }

    const originalBtnText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>⏳ Processing profiles and extracting embeddings...</span>`;

    const formData = new FormData();
    formData.append('name', name);
    formData.append('email', email);
    
    formData.append('files', state.activeSession.selectedFrames['front'], 'front.jpg');
    formData.append('files', state.activeSession.selectedFrames['left'], 'left.jpg');
    formData.append('files', state.activeSession.selectedFrames['right'], 'right.jpg');

    try {
        const response = await fetch(REGISTRATION_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            alert(`Student ${name} and assigned roll_number: ${result.roll_num} is registered successfully`);
            resetScanState();
            document.getElementById('reg-name').value = '';
            document.getElementById('reg-email').value = '';
            backToHome();
        } else {
            const errText = await response.text();
            alert(`Backend Error (${response.status}): ${errText}`);
        }
    } catch (err) {
        alert(`Failed to connect to backend engine: ${err.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
    }
}
window.submitRegistration = submitRegistration;

/**
 * Attendance Sub-Module: Video Handlers & Upload Management Sequences
 */
function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    state.attendanceFile = file;
    
    const nameLabel = document.getElementById('uploaded-video-name');
    nameLabel.innerText = `Selected file: ${file.name}`;
    nameLabel.classList.remove('view-hidden');

    const videoPreview = document.getElementById('attendance-preview');
    videoPreview.src = URL.createObjectURL(file);
    videoPreview.classList.remove('view-hidden');

    document.getElementById('btn-run-attendance').classList.remove('view-hidden');
    document.getElementById('attendance-results').classList.add('view-hidden');
}
window.handleVideoUpload = handleVideoUpload;

async function runAttendanceAnalysis() {
    if (!state.attendanceFile) return;

    const runBtn = document.getElementById('btn-run-attendance');
    const originalBtnText = runBtn.innerHTML;
    
    runBtn.disabled = true;
    runBtn.innerHTML = `<span>⏳ Analyzing frames... This may take a moment depending on video length.</span>`;
    
    document.getElementById('attendance-results').classList.add('hidden');

    const formData = new FormData();
    formData.append('video', state.attendanceFile, state.attendanceFile.name);

    try {
        const response = await fetch(ATTENDANCE_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            displayAttendanceResults(result);
        } else {
            alert(`Backend Error (${response.status}) processing verification pipeline.`);
        }
    } catch (err) {
        alert(`Failed to communicate with backend service module: ${err.message}`);
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = originalBtnText;
    }
}
window.runAttendanceAnalysis = runAttendanceAnalysis;

/**
 * Tabulates data matrix payload parsing back to equivalent tabular structures
 */
function displayAttendanceResults(result) {
    document.getElementById('attendance-results').classList.remove('view-hidden');
    
    const tableHead = document.querySelector('#results-table thead');
    const tableBody = document.querySelector('#results-table tbody');
    const emptyWarn = document.getElementById('attendance-empty-warn');
    
    tableHead.innerHTML = '';
    tableBody.innerHTML = '';
    emptyWarn.classList.add('view-hidden');

    const students = result.students || [];

    if (students.length === 0) {
        emptyWarn.classList.remove('view-hidden');
        document.getElementById('results-table').parentElement.classList.add('view-hidden');
        return;
    }

    document.getElementById('results-table').parentElement.classList.remove('view-hidden');

    const headers = Object.keys(students[0]);
    
    const trHead = document.createElement('tr');
    headers.forEach(h => {
        const th = document.createElement('th');
        th.className = "px-6 py-3 font-semibold text-gray-700 uppercase tracking-wider text-xs border-b border-gray-200 bg-gray-50";
        th.innerText = h.replace('_', ' ');
        trHead.appendChild(th);
    });
    tableHead.appendChild(trHead);

    students.forEach(student => {
        const trRow = document.createElement('tr');
        trRow.className = "hover:bg-gray-50 transition";
        headers.forEach(h => {
            const td = document.createElement('td');
            td.className = "px-6 py-4 white-space-nowrap text-gray-600 border-b border-gray-100";
            td.innerText = student[h] !== null ? student[h] : 'N/A';
            trRow.appendChild(td);
        });
        tableBody.appendChild(trRow);
    });
}