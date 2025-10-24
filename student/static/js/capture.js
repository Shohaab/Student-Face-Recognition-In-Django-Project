const openCameraBtn = document.getElementById("openCamera");
const captureBtn = document.getElementById("capture");
const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const imageInput = document.getElementById("captured_image");

openCameraBtn.addEventListener("click", async () => {
  video.classList.remove("d-none");
  captureBtn.classList.remove("d-none");

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
});

captureBtn.addEventListener("click", () => {
  const context = canvas.getContext("2d");
  canvas.classList.remove("d-none");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const imageData = canvas.toDataURL("image/png");
  imageInput.value = imageData;

  // Stop the video stream after capturing
  video.srcObject.getTracks().forEach(track => track.stop());
  video.classList.add("d-none");
  captureBtn.classList.add("d-none");
});
