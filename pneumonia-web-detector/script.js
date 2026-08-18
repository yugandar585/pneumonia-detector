let model;

async function loadModel() {
  try {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("✅ Model loaded successfully in browser!");
    document.getElementById('result').innerHTML = "Model ready. Upload an X-ray.";
  } catch (error) {
    console.error(error);
    document.getElementById('result').innerHTML = "Error loading model. Check tfjs_model folder.";
  }
}

// Preprocess image (224x224, same as your model)
async function preprocessImage(imageElement) {
  const tensor = tf.browser.fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .div(255.0)
    .expandDims(0);
  return tensor;
}

async function predict(imageElement) {
  if (!model) {
    alert("Model not loaded yet!");
    return;
  }

  const inputTensor = await preprocessImage(imageElement);
  const prediction = await model.predict(inputTensor);
  const score = prediction.dataSync()[0];   // probability of Pneumonia

  const resultDiv = document.getElementById('result');

  if (score > 0.5) {
    resultDiv.innerHTML = `🟥 <span style="color:red;">PNEUMONIA DETECTED</span><br>Confidence: ${(score * 100).toFixed(1)}%`;
    resultDiv.style.color = "red";
  } else {
    resultDiv.innerHTML = `🟩 <span style="color:green;">NORMAL</span><br>Confidence: ${((1 - score) * 100).toFixed(1)}%`;
    resultDiv.style.color = "green";
  }

  inputTensor.dispose(); // Clean memory
}

// Handle file upload
document.getElementById('fileInput').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async function(e) {
    const img = new Image();
    img.src = e.target.result;
    img.onload = async function() {
      document.getElementById('preview').innerHTML = `<img src="${img.src}" alt="X-ray">`;
      await predict(img);
    }
  };
  reader.readAsDataURL(file);
});

// Camera support (basic)
async function useCamera() {
  alert("Camera support coming in next step. For now, use file upload.");
  // We will improve this later with live camera
}

// Load model when page opens
window.onload = loadModel;