<template>
  <div v-if="imageUrl" class="mt-6 relative flex-1">
    <h3 class="text-lg font-semibold mb-2">Select Reference Points on Image</h3>
    <div class="relative inline-block border">
      <img
        ref="imgRef"
        :src="imageUrl"
        alt="First marked frame"
        style="max-width: 100%; display: block"
        @click="handleClick"
      />
      <!-- SVG overlay for points & line -->
      <svg
        v-if="points.length > 0"
        :width="imgWidth"
        :height="imgHeight"
        style="position: absolute; top: 0; left: 0; pointer-events: none"
      >
        <circle
          v-for="(pt, i) in points"
          :key="i"
          :cx="pt.x"
          :cy="pt.y"
          r="6"
          fill="red"
          stroke="white"
          stroke-width="2"
        />
        <line
          v-if="points.length === 2"
          :x1="points[0].x"
          :y1="points[0].y"
          :x2="points[1].x"
          :y2="points[1].y"
          stroke="red"
          stroke-width="3"
        />
        <!-- Distance text -->
        <text
          v-if="points.length === 2"
          :x="(points[0].x + points[1].x) / 2 + 10"
          :y="(points[0].y + points[1].y) / 2"
          fill="red"
          font-weight="bold"
          font-size="16"
        >
          {{ pixelDistance.toFixed(1) }} px
        </text>
      </svg>
    </div>

    <div class="mt-4 flex items-center gap-4">
      <UButton
        :disabled="points.length === 0"
        color="error"
        @click="clearPoints"
      >
        Clear Points
      </UButton>

      <label class="flex items-center gap-2">
        <span>Known distance (mm):</span>
        <UInput
          v-model="mmValue"
          type="number"
          min="0"
          step="0.01"
          :disabled="points.length !== 2"
        />
      </label>

      <button
        class="bg-blue-500 text-white px-4 py-1 rounded"
        :disabled="!canSubmit"
        @click="submitPoints"
      >
        Submit
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from "vue";

const props = defineProps({
  imageUrl: String,
  videoName: String,
});

const emit = defineEmits(["submit"]);

const points = ref([]);
const mmValue = ref(null);
const imgRef = ref(null);
const imgWidth = ref(0);
const imgHeight = ref(0);

// Update image dimensions on load
const onImageLoad = () => {
  if (!imgRef.value) return;
  imgWidth.value = imgRef.value.naturalWidth;
  imgHeight.value = imgRef.value.naturalHeight;
};
watch(imgRef, () => {
  if (imgRef.value) {
    imgRef.value.onload = onImageLoad;
  }
});

const clearPoints = () => {
  points.value = [];
  mmValue.value = null;
};

// Handle click, enforce same x coordinate as first point
const handleClick = (event) => {
  if (points.value.length >= 2) return;

  const rect = imgRef.value.getBoundingClientRect();
  let x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  if (points.value.length === 1) {
    // Align second point's x to first point's x
    x = points.value[0].x;
  }

  points.value.push({ x, y });
};

// Compute pixel distance between two points (vertical distance only)
const pixelDistance = computed(() => {
  if (points.value.length !== 2) return 0;
  return Math.abs(points.value[1].y - points.value[0].y);
});

const canSubmit = computed(() => {
  return points.value.length === 2 && mmValue.value > 0;
});

const submitPoints = () => {
  if (!canSubmit.value) return;
  // Emit event with points, mmValue, and videoName for backend API call
  emit("submit", {
    x1: points.value[0].x,
    y1: points.value[0].y,
    x2: points.value[1].x,
    y2: points.value[1].y,
    mmValue: mmValue.value,
    videoName: props.videoName,
  });
};
</script>

<style scoped>
/* Optional: cursor pointer on image */
img {
  cursor: crosshair;
}
</style>
