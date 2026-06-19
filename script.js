// ===== Configuration =====
// 1. Add these configuration lines at the VERY top of script.js
ort.env.wasm.numThreads = 1; 
ort.env.wasm.proxy = true; // Runs the model in a web worker to prevent UI freeze

async function loadModel() {
  try {
    modelStatus.textContent = '🟡 Loading...';
    console.log("Attempting to load model from:", MODEL_PATH);

    // 2. Create session with explicit settings
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    console.log('✅ Model Loaded Successfully');
    modelStatus.textContent = '✅ Model Ready';
    modelStatus.className = 'status-badge ready';
  } catch (e) {
    console.error('Model Load Failed Details:', e);
    modelStatus.textContent = '❌ Model Load Failed';
    modelStatus.className = 'status-badge error';
    
    // Check if it's a 404 error
    if (e.message.includes('404') || e.message.includes('fetch')) {
        alert("Model file not found! Check if 'model/best.onnx' exists in your GitHub folder.");
    } else {
        alert("Model Error: " + e.message);
    }
  }
}
loadModel();
const MODEL_PATH = 'model/best.onnx';
const INPUT_SIZE = 640;
// IMPORTANT: Update these class names to match your training!
const CLASSES = ['person', 'id_card']; // Change based on your model
const PERSON_CLASS = 'person';
const ID_CLASS = 'id_card';

// ===== Elements =====
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const alarm = document.getElementById('alarm');
const modelStatus = document.getElementById('modelStatus');
const alertBox = document.getElementById('alertBox');
const alertText = document.getElementById('alertText');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const soundToggle = document.getElementById('soundToggle');

let session = null;
let currentStream = null;
let animationId = null;
let mode = null;

confSlider.oninput = () => confValue.textContent = confSlider.value;

// ===== Load Model =====
async function loadModel() {
  try {
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    modelStatus.textContent = '✅ Model Ready';
    modelStatus.className = 'status-badge ready';
  } catch (e) {
    modelStatus.textContent = '❌ Model Load Failed';
    modelStatus.className = 'status-badge error';
    console.error(e);
  }
}
loadModel();

// ===== Preprocessing =====
function preprocess(source, w, h) {
  const tmp = document.createElement('canvas');
  tmp.width = INPUT_SIZE; tmp.height = INPUT_SIZE;
  const tctx = tmp.getContext('2d');
  tctx.fillStyle = '#727272';
  tctx.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);
  const r = Math.min(INPUT_SIZE/w, INPUT_SIZE/h);
  const nw = w*r, nh = h*r;
  const dx = (INPUT_SIZE-nw)/2, dy=(INPUT_SIZE-nh)/2;
  tctx.drawImage(source, dx, dy, nw, nh);
  const imgData = tctx.getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data;
  const float = new Float32Array(3*INPUT_SIZE*INPUT_SIZE);
  for (let i=0;i<INPUT_SIZE*INPUT_SIZE;i++){
    float[i] = imgData[i*4]/255;
    float[i+INPUT_SIZE*INPUT_SIZE] = imgData[i*4+1]/255;
    float[i+2*INPUT_SIZE*INPUT_SIZE] = imgData[i*4+2]/255;
  }
  return { tensor: new ort.Tensor('float32', float, [1,3,INPUT_SIZE,INPUT_SIZE]),
           ratio: r, dx, dy };
}

// ===== Postprocessing =====
function postprocess(output, ratio, dx, dy, confThresh) {
  // YOLOv8 output: [1, 4+nc, 8400]
  const data = output.data;
  const dims = output.dims;
  const nc = dims[1] - 4;
  const np = dims[2];
  const boxes = [];
  for (let i=0;i<np;i++){
    let maxC = 0, cls = 0;
    for (let j=0;j<nc;j++){
      const s = data[(4+j)*np + i];
      if (s>maxC){ maxC=s; cls=j; }
    }
    if (maxC < confThresh) continue;
    const cx = data[i], cy=data[np+i], w=data[2*np+i], h=data[3*np+i];
    const x = (cx - w/2 - dx)/ratio;
    const y = (cy - h/2 - dy)/ratio;
    boxes.push({ x, y, w:w/ratio, h:h/ratio, score:maxC, class:cls });
  }
  return nms(boxes, 0.45);
}

function nms(boxes, iouThresh){
  boxes.sort((a,b)=>b.score-a.score);
  const kept=[];
  while(boxes.length){
    const b=boxes.shift(); kept.push(b);
    boxes = boxes.filter(x=>iou(b,x)<iouThresh);
  }
  return kept;
}
function iou(a,b){
  const x1=Math.max(a.x,b.x), y1=Math.max(a.y,b.y);
  const x2=Math.min(a.x+a.w,b.x+b.w), y2=Math.min(a.y+a.h,b.y+b.h);
  const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);
  return inter/(a.w*a.h+b.w*b.h-inter);
}

// ===== Draw + Logic =====
function drawDetections(boxes){
  let persons=0, ids=0, violations=0;
  const personBoxes = boxes.filter(b=>CLASSES[b.class]===PERSON_CLASS);
  const idBoxes = boxes.filter(b=>CLASSES[b.class]===ID_CLASS);
  persons = personBoxes.length;
  ids = idBoxes.length;

  // Check each person for an ID overlap
  personBoxes.forEach(p=>{
    const hasID = idBoxes.some(id=>{
      const cx = id.x + id.w/2, cy = id.y + id.h/2;
      return cx>=p.x && cx<=p.x+p.w && cy>=p.y && cy<=p.y+p.h;
    });
    p.violation = !hasID;
    if(!hasID) violations++;
  });

  // Draw boxes
  boxes.forEach(b=>{
    const label = CLASSES[b.class] || `cls${b.class}`;
    let color = '#00ff88';
    if (label===PERSON_CLASS) color = b.violation ? '#ff3344' : '#00ff88';
    else if (label===ID_CLASS) color = '#00d4ff';

    ctx.strokeStyle = color; ctx.lineWidth = 3;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.fillStyle = color;
    const txt = `${label} ${(b.score*100).toFixed(0)}%`;
    ctx.font = 'bold 16px sans-serif';
    const tw = ctx.measureText(txt).width;
    ctx.fillRect(b.x, b.y-22, tw+10, 22);
    ctx.fillStyle = '#000';
    ctx.fillText(txt, b.x+5, b.y-6);
  });

  document.getElementById('personCount').textContent = persons;
  document.getElementById('idCount').textContent = ids;
  document.getElementById('violationCount').textContent = violations;

  if (violations>0){
    alertBox.classList.add('violation');
    alertText.textContent = `⚠️ ${violations} person(s) without ID!`;
    if (soundToggle.checked && alarm.paused) alarm.play().catch(()=>{});
  } else {
    alertBox.classList.remove('violation');
    alertText.textContent = '✅ All Clear';
    if (!alarm.paused){ alarm.pause(); alarm.currentTime=0; }
  }
}

async function detect(source, w, h){
  if (!session) return;
  const { tensor, ratio, dx, dy } = preprocess(source, w, h);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const boxes = postprocess(out[session.outputNames[0]], ratio, dx, dy, parseFloat(confSlider.value));
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  drawDetections(boxes);
}

// ===== Input Handlers =====
document.getElementById('btnImage').onclick = ()=>document.getElementById('imageInput').click();
document.getElementById('btnVideo').onclick = ()=>document.getElementById('videoInput').click();
document.getElementById('btnWebcam').onclick = startWebcam;
document.getElementById('btnStop').onclick = stopAll;

document.getElementById('imageInput').onchange = async (e)=>{
  stopAll();
  const file = e.target.files[0]; if(!file) return;
  const img = new Image();
  img.onload = async ()=>{
    canvas.width = img.width; canvas.height = img.height;
    overlay.style.display='none';
    await detect(img, img.width, img.height);
  };
  img.src = URL.createObjectURL(file);
};

document.getElementById('videoInput').onchange = (e)=>{
  stopAll();
  const file = e.target.files[0]; if(!file) return;
  video.src = URL.createObjectURL(file);
  video.hidden = false; video.controls = true; video.play();
  mode = 'video';
  video.onloadeddata = ()=>{
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    overlay.style.display='none';
    processVideo();
  };
};

async function startWebcam(){
  stopAll();
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({ video:true });
    video.srcObject = currentStream;
    await video.play();
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    overlay.style.display='none';
    mode = 'webcam';
    processVideo();
  } catch(e){ alert('Webcam access denied'); }
}

async function processVideo(){
  if (!video.paused && !video.ended){
    await detect(video, video.videoWidth, video.videoHeight);
  }
  animationId = requestAnimationFrame(processVideo);
}

function stopAll(){
  if (animationId) cancelAnimationFrame(animationId);
  if (currentStream){ currentStream.getTracks().forEach(t=>t.stop()); currentStream=null; }
  video.pause(); video.srcObject=null; video.src=''; video.hidden=true;
  alarm.pause(); alarm.currentTime=0;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  overlay.style.display='flex';
  alertBox.classList.remove('violation');
  alertText.textContent = '✅ All Clear';
}
