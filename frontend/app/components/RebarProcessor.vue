<template>
  <div
    class="px-4 pt-4 pb-10 w-full min-h-screen flex flex-col items-center *:flex-1 gap-4"
  >
    <div class="mb-4 flex gap-4 h-12">
      <UButton
        :class="{ 'font-bold underline': viewMode === 'upload' }"
        @click="viewMode = 'upload'"
      >
        Upload & Process
      </UButton>
      <UButton
        :class="{ 'font-bold underline': viewMode === 'list' }"
        @click="viewMode = 'list'"
      >
        Processed Videos
      </UButton>
    </div>
    <div v-if="viewMode === 'upload'" class="flex-1 py-5">
      <h2 class="text-xl font-bold mb-4">Upload Video</h2>
      <input
        type="file"
        accept="video/*"
        class="mb-4 border-green-300"
        @change="handleFile"
      />
      <div v-if="videoUrl" class="mb-4">
        <video
          ref="videoPlayerRef"
          :src="videoUrl"
          controls
          class="w-full max-w-md border"
          @timeupdate="updateCurrentFrame"
        />
        <div class="mt-2">
          Current time: {{ currentTime.toFixed(2) }}s<br />
          Current frame: {{ currentFrame }}
        </div>
      </div>

      <div class="mb-4">
        <label class="mr-2">FPS:</label>
        <input
          v-model.number="FPS"
          type="number"
          min="1"
          step="any"
          class="border px-2 py-1 w-20"
          placeholder="30"
        />
      </div>

      <div class="mb-4">
        <label class="mr-2">Skip Start Frames:</label>
        <input
          v-model.number="skipStart"
          type="number"
          min="0"
          class="border px-2 py-1 w-20"
        />
      </div>
      <div class="mb-4">
        <label class="mr-2">Skip End Frames:</label>
        <input
          v-model.number="skipEnd"
          type="number"
          min="0"
          class="border px-2 py-1 w-20"
        />
      </div>

      <button
        :disabled="!video || processing"
        class="bg-blue-500 text-white px-4 py-2 rounded mr-2"
        @click="startProcessing"
      >
        Start
      </button>

      <button
        :disabled="!processing"
        class="bg-red-600 text-white px-4 py-2 rounded"
        @click="stopProcessing"
      >
        Stop
      </button>
    </div>
    <div v-else>
      <ProcessedVideosList @select="onVideoSelect" @delete="onDelete" />
    </div>

    <div v-if="plotUrl" class="mt-6">
      <h3 class="text-lg font-semibold mb-2">Elongation Plot</h3>
      <img :src="plotUrl" alt="Elongation Plot" class="border" />
    </div>
    <div class="flex-1 relative z-20">
      <PixelToMmSelector
        v-if="firstMarkedImageUrl"
        :image-url="firstMarkedImageUrl"
        :video-name="videoName"
        @submit="handlePixelToMmSubmit"
      />
    </div>
    <div v-if="newPlotUrl" :key="counter" class="mt-6">
      <h3 class="text-lg font-semibold mb-2">Elongation Plot (mm)</h3>
      <img :src="newPlotUrl" alt="Elongation Plot (mm)" class="border" />
    </div>
    <div
      v-if="logs.length"
      class="my-5 p-2 min-h-40 max-h-64 overflow-y-auto text-sm flex-1"
    >
      <div v-for="(log, i) in logs" :key="i">{{ log }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
const videoName = ref("");
const firstMarkedImageUrl = ref(null);
const video = ref(null);
const viewMode = ref("upload"); // or 'list'
const skipStart = ref(0);
const skipEnd = ref(0);
const videoUrl = ref(null);
const currentTime = ref(0);
const currentFrame = ref(0);
const counter = ref(0); // Used to force re-render of new mm plot
const FPS = ref(30); // Default FPS value
const logs = ref([]);
const newPlotUrl = ref(null);
const plotUrl = ref(null);
const processing = ref(false);
const safeVideoName = computed(() => {
  const cleaned = videoName.value.replace(/ /g, "_");
  return cleaned.replace(/\.(mp4|mov|avi|mkv)$/i, "");
});

const videoPlayerRef = ref(null);

const onDelete = (name) => {
  if (videoName.value === name) {
    firstMarkedImageUrl.value = null;
    videoName.value = "";
    plotUrl.value = null;
    newPlotUrl.value = null; // clear new mm plot
  }
};

const loadFirstMarkedImage = (name, selected = false) => {
  videoName.value = name;

  firstMarkedImageUrl.value = `/backend/first_marked_image?video_name=${encodeURIComponent(
    safeVideoName.value
  )}`;
  plotUrl.value = `/backend/results/elongation_plot_${encodeURIComponent(
    safeVideoName.value
  )}.png`;
  if (selected) {
    // If this video is selected, also load the new mm plot URL
    newPlotUrl.value = `/backend/results/elongation_plot_mm_${encodeURIComponent(
      safeVideoName.value
    )}.png`;
  } else {
    newPlotUrl.value = null; // reset new mm plot URL
  }
};
const onVideoSelect = (name) => {
  loadFirstMarkedImage(name, true);
  viewMode.value = "upload";
};

const handleFile = (e) => {
  if (videoUrl.value) {
    URL.revokeObjectURL(videoUrl.value);
  }
  video.value = e.target.files[0];
  videoUrl.value = URL.createObjectURL(video.value);
  currentFrame.value = 0;
  currentTime.value = 0;
};

const updateCurrentFrame = () => {
  const videoPlayer = videoPlayerRef.value;
  if (!videoPlayer || !FPS.value) return;
  currentTime.value = videoPlayer.currentTime;
  currentFrame.value = Math.floor(currentTime.value * FPS.value);
};

const startProcessing = async () => {
  logs.value = [];
  plotUrl.value = null;
  processing.value = true;
  firstMarkedImageUrl.value = null;

  const formData = new FormData();
  formData.append("video", video.value);
  formData.append("skip_start", skipStart.value);
  formData.append("skip_end", skipEnd.value);
  try {
    const res = await fetch("/backend/process/", {
      method: "POST",
      body: formData,
    });
    const json = await res.json();

    if (res.ok) {
      logs.value.push("✅ Processing finished.");
      plotUrl.value = `/backend/${json.plot}`;
      loadFirstMarkedImage(video.value.name);
    } else {
      logs.value.push("❌ Error: " + (json.error || "Unknown error"));
    }
  } catch (err) {
    logs.value.push("❌ Error: " + err.message);
  } finally {
    processing.value = false;
  }
};

const stopProcessing = async () => {
  try {
    await fetch("/backend/stop/", {
      method: "POST",
    });
    logs.value.push("🛑 Stop requested.");
  } catch (err) {
    logs.value.push("❌ Stop error: " + err.message);
  }
};
const handlePixelToMmSubmit = async (data) => {
  try {
    const res = await fetch("/backend/pixel_to_mm/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await res.json();
    console.log("Pixel-to-mm result:", result);

    if (res.ok && result.plot) {
      // Update reactive variable to show the new plot
      newPlotUrl.value = null; // Reset to force re-render
      await new Promise((resolve) => setTimeout(resolve, 100)); // Small delay to ensure reactivity
      newPlotUrl.value = `/backend/${result.plot}`;
      counter.value++; // Increment counter to force re-render
    } else {
      console.error(
        "Failed to generate pixel to mm plot:",
        result.error || "Unknown error"
      );
    }
  } catch (err) {
    console.error("Error calling pixel_to_mm API:", err);
  }
};
onMounted(() => {
  const ws = new WebSocket("ws://localhost:8000/ws");
  ws.onmessage = (e) => logs.value.push(e.data);
});
</script>
