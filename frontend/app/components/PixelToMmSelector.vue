<template>
  <div v-if="imageUrl" class="space-y-6">
    <div class="border-b border-gray-600 pb-4">
      <h3 class="text-lg font-semibold text-white">
        Calibrate Pixel to Millimeter Conversion
      </h3>
      <p class="mt-1 text-sm text-gray-400">
        Click on two points on the image to measure a known distance. This will
        help convert pixel measurements to millimeters.
      </p>
    </div>

    <!-- Image Selection Area -->
    <div class="bg-gray-900 rounded-lg p-4 border border-gray-700">
      <div
        class="relative inline-block border-2 border-gray-600 rounded-lg overflow-hidden"
      >
        <img
          ref="imgRef"
          :src="imageUrl"
          alt="First marked frame"
          class="max-w-full h-auto block cursor-crosshair"
          @click="handleClick"
          @load="onImageLoad"
        />

        <!-- SVG overlay for points & line -->
        <svg
          v-if="points.length > 0"
          :width="imgWidth"
          :height="imgHeight"
          class="absolute top-0 left-0 pointer-events-none"
        >
          <!-- Points -->
          <circle
            v-for="(pt, i) in points"
            :key="i"
            :cx="pt.x"
            :cy="pt.y"
            r="8"
            :fill="i === 0 ? '#3B82F6' : '#EF4444'"
            stroke="white"
            stroke-width="3"
          />

          <!-- Point labels -->
          <text
            v-for="(pt, i) in points"
            :key="`label-${i}`"
            :x="pt.x + 15"
            :y="pt.y - 10"
            fill="white"
            font-weight="bold"
            font-size="14"
            class="drop-shadow-lg"
          >
            {{ i === 0 ? "Point 1" : "Point 2" }}
          </text>

          <!-- Line between points -->
          <line
            v-if="points.length === 2"
            :x1="points[0].x"
            :y1="points[0].y"
            :x2="points[1].x"
            :y2="points[1].y"
            stroke="#10B981"
            stroke-width="4"
            stroke-dasharray="5,5"
          />

          <!-- Distance measurement -->
          <g v-if="points.length === 2">
            <rect
              :x="(points[0].x + points[1].x) / 2 - 40"
              :y="(points[0].y + points[1].y) / 2 - 15"
              width="80"
              height="30"
              fill="#374151"
              rx="4"
            />
            <text
              :x="(points[0].x + points[1].x) / 2"
              :y="(points[0].y + points[1].y) / 2 + 5"
              fill="white"
              font-weight="bold"
              font-size="12"
              text-anchor="middle"
            >
              {{ pixelDistance.toFixed(1) }} px
            </text>
          </g>
        </svg>
      </div>

      <!-- Instructions -->
      <div class="mt-4 text-sm text-gray-400">
        <div class="flex items-center space-x-2 mb-2">
          <div class="w-3 h-3 bg-blue-500 rounded-full" />
          <span>Click to place first point</span>
        </div>
        <div class="flex items-center space-x-2 mb-2">
          <div class="w-3 h-3 bg-red-500 rounded-full" />
          <span>Click to place second point (same X position)</span>
        </div>
        <div class="flex items-center space-x-2">
          <div class="w-3 h-3 bg-green-500 rounded-full" />
          <span>Line shows vertical distance measurement</span>
        </div>
      </div>
    </div>

    <!-- Controls -->
    <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Action Buttons -->
        <div class="flex space-x-3">
          <button
            :disabled="points.length === 0"
            class="flex items-center px-4 py-2 text-sm font-medium text-red-400 bg-red-900 rounded-md hover:bg-red-800 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
            @click="clearPoints"
          >
            <svg
              class="w-4 h-4 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
            Clear Points
          </button>
        </div>

        <!-- Distance Input -->
        <div class="flex items-center space-x-3">
          <label class="text-sm font-medium text-gray-300 whitespace-nowrap">
            Known distance:
          </label>
          <div class="flex items-center space-x-2">
            <input
              v-model="mmValue"
              type="number"
              min="0"
              step="0.01"
              :disabled="points.length !== 2"
              class="w-24 px-3 py-2 border border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white disabled:bg-gray-800 disabled:cursor-not-allowed"
              placeholder="0.00"
            />
            <span class="text-sm text-gray-400">mm</span>
          </div>
        </div>
      </div>

      <!-- Submit Button -->
      <div class="mt-4 pt-4 border-t border-gray-600">
        <button
          class="w-full flex items-center justify-center px-4 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="!canSubmit"
          @click="submitPoints"
        >
          <svg
            class="w-5 h-5 mr-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          {{
            canSubmit
              ? "Generate Millimeter Plot"
              : "Select two points and enter distance"
          }}
        </button>
      </div>

      <!-- Conversion Info -->
      <div
        v-if="points.length === 2 && mmValue > 0"
        class="mt-4 p-3 bg-blue-900 rounded-md border border-blue-700"
      >
        <div class="flex items-center justify-between text-sm">
          <span class="text-blue-300">Conversion ratio:</span>
          <span class="font-mono font-medium text-blue-200">
            {{ (mmValue / pixelDistance).toFixed(4) }} mm/pixel
          </span>
        </div>
      </div>
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
/* Custom cursor for image */
img {
  cursor: crosshair;
}
</style>
