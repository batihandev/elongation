<template>
  <div class="min-h-screen bg-gray-900">
    <!-- Header -->
    <header class="bg-gray-800 shadow-lg border-b border-gray-700">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <h1 class="text-xl font-semibold text-white">
            Rebar Elongation Processor
          </h1>
          <div class="flex items-center space-x-4">
            <span class="text-sm text-gray-400 hidden sm:inline"
              >Processing Status:</span
            >
            <div class="flex items-center">
              <div
                :class="[
                  'w-2 h-2 rounded-full mr-2',
                  processing ? 'bg-yellow-400 animate-pulse' : 'bg-green-400',
                ]"
              />
              <span class="text-sm font-medium text-gray-300 hidden sm:inline">
                {{ processing ? "Processing..." : "Ready" }}
              </span>
            </div>
            <!-- Hamburger menu for mobile -->
            <button
              class="sm:hidden ml-4 p-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              aria-label="Open navigation menu"
              @click="openSidebar"
            >
              <svg
                class="w-7 h-7 text-gray-200"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </header>

    <div class="max-w-7xl mx-auto px-2 sm:px-6 lg:px-8 py-4 sm:py-6">
      <div class="flex gap-2 sm:gap-6">
        <!-- Sidebar Navigation (desktop) -->
        <div class="w-64 flex-shrink-0 hidden sm:block">
          <nav
            class="bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-4"
          >
            <h2 class="text-lg font-semibold text-white mb-4">Navigation</h2>
            <div class="space-y-2">
              <button
                :class="[
                  'w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  viewMode === 'upload'
                    ? 'bg-blue-600 text-white border border-blue-500'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white',
                ]"
                @click="viewMode = 'upload'"
              >
                <div class="flex items-center">
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
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  Upload & Process
                </div>
              </button>
              <button
                :class="[
                  'w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  viewMode === 'list'
                    ? 'bg-blue-600 text-white border border-blue-500'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white',
                ]"
                @click="viewMode = 'list'"
              >
                <div class="flex items-center">
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
                      d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                    />
                  </svg>
                  Processed Videos
                </div>
              </button>
            </div>
          </nav>
        </div>

        <!-- Sidebar Navigation (mobile overlay) -->
        <transition name="fade">
          <div v-if="sidebarOpen" class="fixed inset-0 z-50 flex">
            <div class="fixed inset-0 bg-black/40" @click="closeSidebar" />
            <nav
              class="relative w-64 max-w-full border-r border-gray-700 shadow-xl h-full flex flex-col p-4 z-50 bg-gray-800"
            >
              <div class="flex items-center justify-between mb-4">
                <h2 class="text-lg font-semibold text-white">Navigation</h2>
                <button
                  aria-label="Close navigation menu"
                  class="p-2 rounded hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  @click="closeSidebar"
                >
                  <svg
                    class="w-6 h-6 text-gray-200"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
              <div class="space-y-2">
                <button
                  :class="[
                    'w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors',
                    viewMode === 'upload'
                      ? 'bg-blue-600 text-white border border-blue-500'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white',
                  ]"
                  @click="
                    viewMode = 'upload';
                    closeSidebar();
                  "
                >
                  <div class="flex items-center">
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
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                    Upload & Process
                  </div>
                </button>
                <button
                  :class="[
                    'w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors',
                    viewMode === 'list'
                      ? 'bg-blue-600 text-white border border-blue-500'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white',
                  ]"
                  @click="
                    viewMode = 'list';
                    closeSidebar();
                  "
                >
                  <div class="flex items-center">
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
                        d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                      />
                    </svg>
                    Processed Videos
                  </div>
                </button>
              </div>
            </nav>
          </div>
        </transition>

        <!-- Main Content -->
        <div class="flex-1 min-w-0">
          <!-- Upload & Process Section -->
          <div v-if="viewMode === 'upload'" class="space-y-6">
            <div
              class="bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-6"
            >
              <h2 class="text-lg font-semibold text-white mb-4">
                Upload Video
              </h2>

              <!-- File Upload Area -->
              <div class="mb-6">
                <div
                  class="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-blue-500 transition-colors bg-gray-900"
                  @click="$refs.fileInput.click()"
                  @dragover.prevent
                  @drop.prevent="handleDrop"
                >
                  <svg
                    class="mx-auto h-12 w-12 text-gray-500"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </svg>
                  <div class="mt-4">
                    <p class="text-sm text-gray-400">
                      <span
                        class="font-medium text-blue-400 hover:text-blue-300 cursor-pointer"
                        >Click to upload</span
                      >
                      or drag and drop
                    </p>
                    <p class="text-xs text-gray-500">
                      MP4, MOV, AVI, MKV up to 100MB
                    </p>
                  </div>
                  <input
                    ref="fileInput"
                    type="file"
                    accept="video/*"
                    class="hidden"
                    @change="handleFile"
                  />
                </div>
                <div
                  v-if="video"
                  class="mt-4 p-3 bg-green-900 border border-green-700 rounded-md"
                >
                  <div class="flex items-center">
                    <svg
                      class="w-5 h-5 text-green-400 mr-2"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fill-rule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clip-rule="evenodd"
                      />
                    </svg>
                    <span class="text-sm text-green-300">{{ video.name }}</span>
                  </div>
                </div>
              </div>

              <!-- Video Preview -->
              <div v-if="videoUrl" class="mb-6">
                <h3 class="text-md font-medium text-white mb-3">
                  Video Preview
                </h3>
                <div class="relative">
                  <video
                    ref="videoPlayerRef"
                    :src="videoUrl"
                    controls
                    class="w-full max-w-md rounded-lg border border-gray-600 shadow-lg"
                    @timeupdate="updateCurrentFrame"
                  />
                  <div class="mt-3 text-sm text-gray-400">
                    <div class="flex justify-between">
                      <span>Current time: {{ currentTime.toFixed(2) }}s</span>
                      <span>Frame: {{ currentFrame }}</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Processing Settings -->
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <label class="block text-sm font-medium text-gray-300 mb-1"
                    >FPS</label
                  >
                  <input
                    v-model.number="FPS"
                    type="number"
                    min="1"
                    step="any"
                    class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white"
                    placeholder="30"
                  />
                </div>
                <div>
                  <label class="block text-sm font-medium text-gray-300 mb-1"
                    >Skip Start Frames</label
                  >
                  <input
                    v-model.number="skipStart"
                    type="number"
                    min="0"
                    class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white"
                  />
                </div>
                <div>
                  <label class="block text-sm font-medium text-gray-300 mb-1"
                    >Skip End Frames</label
                  >
                  <input
                    v-model.number="skipEnd"
                    type="number"
                    min="0"
                    class="w-full px-3 py-2 border border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500 bg-gray-700 text-white"
                  />
                </div>
              </div>

              <!-- Action Buttons -->
              <div class="flex space-x-3">
                <button
                  :disabled="!video || processing"
                  class="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                  @click="startProcessing"
                >
                  <svg
                    v-if="!processing"
                    class="w-4 h-4 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <svg
                    v-else
                    class="w-4 h-4 mr-2 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      class="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      stroke-width="4"
                    />
                    <path
                      class="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  {{ processing ? "Processing..." : "Start Processing" }}
                </button>
                <button
                  :disabled="!processing"
                  class="flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                  @click="stopProcessing"
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
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                  Stop
                </button>
              </div>
            </div>

            <!-- Results Section -->
            <div
              v-if="plotUrl || newPlotUrl"
              class="bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-6"
            >
              <h3 class="text-lg font-semibold text-white mb-4">Results</h3>

              <!-- Original Plot -->
              <div v-if="plotUrl" class="mb-6">
                <h4 class="text-md font-medium text-white mb-3">
                  Elongation Plot (Pixels)
                </h4>
                <template v-if="!plotImgError">
                  <img
                    :src="plotUrl + '?t=' + Date.now()"
                    alt="Elongation Plot"
                    class="max-w-full h-auto rounded-lg border border-gray-600 shadow-lg"
                    @error="plotImgError = true"
                  />
                </template>
                <div
                  v-else
                  class="flex items-center justify-center h-32 bg-gray-900 border border-gray-700 rounded-lg"
                >
                  <svg
                    class="w-8 h-8 text-gray-500 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12A9 9 0 113 12a9 9 0 0118 0z"
                    />
                  </svg>
                  <span class="text-gray-400">Plot not available</span>
                </div>
              </div>

              <!-- MM Plot -->
              <div v-if="newPlotUrl" class="mb-6">
                <h4 class="text-md font-medium text-white mb-3">
                  Elongation Plot (mm)
                </h4>
                <template v-if="!mmPlotImgError">
                  <img
                    :src="newPlotUrl + '?t=' + Date.now()"
                    alt="Elongation Plot (mm)"
                    class="max-w-full h-auto rounded-lg border border-gray-600 shadow-lg"
                    @error="mmPlotImgError = true"
                  />
                </template>
                <div
                  v-else
                  class="flex items-center justify-center h-32 bg-gray-900 border border-gray-700 rounded-lg"
                >
                  <svg
                    class="w-8 h-8 text-gray-500 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12A9 9 0 113 12a9 9 0 0118 0z"
                    />
                  </svg>
                  <span class="text-gray-400">Plot not available</span>
                </div>
              </div>

              <!-- Data + Downloads -->
              <div
                v-if="finalRows.length"
                class="mt-6 bg-gray-900 rounded-lg border border-gray-700 p-4"
              >
                <div
                  class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4"
                >
                  <h4 class="text-md font-medium text-white">
                    Elongation Data (1 sample / second, monotonic)
                  </h4>
                  <div class="flex flex-wrap gap-2">
                    <!-- Save plot images -->
                    <a
                      v-if="plotUrl"
                      :href="plotUrl"
                      :download="`elongation_plot_${videoName}.png`"
                      class="px-3 py-1.5 text-xs sm:text-sm rounded-md bg-gray-700 hover:bg-gray-600 text-gray-100 border border-gray-500"
                    >
                      Save Plot (px)
                    </a>
                    <a
                      v-if="newPlotUrl"
                      :href="newPlotUrl"
                      :download="`elongation_plot_mm_${videoName}.png`"
                      class="px-3 py-1.5 text-xs sm:text-sm rounded-md bg-gray-700 hover:bg-gray-600 text-gray-100 border border-gray-500"
                    >
                      Save Plot (mm)
                    </a>

                    <!-- CSV downloads -->
                    <a
                      v-if="finalCsvUrl"
                      :href="finalCsvUrl"
                      download
                      class="px-3 py-1.5 text-xs sm:text-sm rounded-md bg-blue-600 hover:bg-blue-700 text-white"
                    >
                      Download Full CSV
                    </a>
                    <a
                      v-if="simplifiedCsvUrl"
                      :href="simplifiedCsvUrl"
                      class="px-3 py-1.5 text-xs sm:text-sm rounded-md bg-blue-600 hover:bg-blue-700 text-white"
                    >
                      Download Simplified CSV
                    </a>
                    <a
                      v-if="pixelToMmCsvUrl"
                      :href="pixelToMmCsvUrl"
                      download
                      class="px-3 py-1.5 text-xs sm:text-sm rounded-md bg-blue-600 hover:bg-blue-700 text-white"
                    >
                      Download Pixel→mm CSV
                    </a>
                  </div>
                </div>

                <!-- Table -->
                <div
                  class="overflow-x-auto max-h-64 border border-gray-700 rounded-md"
                >
                  <table
                    class="min-w-full text-xs sm:text-sm divide-y divide-gray-700"
                  >
                    <thead class="bg-gray-900">
                      <tr>
                        <th
                          class="px-3 py-2 text-left font-medium text-gray-300"
                        >
                          Time (s)
                        </th>
                        <th
                          class="px-3 py-2 text-left font-medium text-gray-300"
                        >
                          Elongation %
                        </th>
                        <th
                          v-if="hasMmData"
                          class="px-3 py-2 text-left font-medium text-gray-300"
                        >
                          Elongation (mm)
                        </th>
                      </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-800 bg-gray-900/50">
                      <tr v-for="(row, idx) in finalRows" :key="idx">
                        <td class="px-3 py-1.5 text-gray-200">
                          {{
                            row.timestamp_s != null
                              ? row.timestamp_s.toFixed(2)
                              : idx
                          }}
                        </td>
                        <td class="px-3 py-1.5 text-gray-200">
                          {{ row.elongation_monotonic?.toFixed(2) }}%
                        </td>
                        <td v-if="hasMmData" class="px-3 py-1.5 text-gray-200">
                          {{
                            row.elongation_monotonic_mm != null
                              ? row.elongation_monotonic_mm.toFixed(3)
                              : "-"
                          }}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <!-- Pixel to MM Selector -->
            <div
              v-if="firstMarkedImageUrl"
              class="bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-6"
            >
              <PixelToMmSelector
                :image-url="firstMarkedImageUrl + '&t=' + Date.now()"
                :video-name="videoName"
                @submit="handlePixelToMmSubmit"
              />
            </div>
          </div>

          <!-- Processed Videos List -->
          <div v-else>
            <ProcessedVideosList @select="onVideoSelect" @delete="onDelete" />
          </div>

          <!-- Processing Logs -->
          <div
            v-if="logs.length"
            class="bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-6 mt-6"
          >
            <h3 class="text-lg font-semibold text-white mb-4">
              Processing Logs
            </h3>
            <div
              class="bg-gray-900 rounded-md p-4 max-h-64 overflow-y-auto border border-gray-700"
            >
              <div
                v-for="(log, i) in logs"
                :key="i"
                class="text-sm font-mono text-gray-300 mb-1"
              >
                {{ log }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";

const videoName = ref("");
const firstMarkedImageUrl = ref(null);
const video = ref(null);
const viewMode = ref("upload");
const skipStart = ref(0);
const skipEnd = ref(0);
const videoUrl = ref(null);
const currentTime = ref(0);
const currentFrame = ref(0);
const counter = ref(0);
const FPS = ref(30);
const logs = ref([]);
const newPlotUrl = ref(null);
const plotUrl = ref(null);
const processing = ref(false);
const plotImgError = ref(false);
const mmPlotImgError = ref(false);
const sidebarOpen = ref(false); // For mobile menu

const videoPlayerRef = ref(null);

const openSidebar = () => {
  sidebarOpen.value = true;
};
const closeSidebar = () => {
  sidebarOpen.value = false;
};

const onDelete = (name) => {
  if (videoName.value === name) {
    firstMarkedImageUrl.value = null;
    videoName.value = "";
    plotUrl.value = null;
    newPlotUrl.value = null;
  }
};

const loadFirstMarkedImage = (baseName, selected = false) => {
  videoName.value = baseName;

  firstMarkedImageUrl.value = `/backend/first_marked_image?base_name=${encodeURIComponent(
    baseName
  )}`;

  plotUrl.value = `/backend/results/elongation_plot_${encodeURIComponent(
    baseName
  )}.png`;

  if (selected) {
    newPlotUrl.value = `/backend/results/elongation_plot_mm_${encodeURIComponent(
      baseName
    )}.png`;
  } else {
    newPlotUrl.value = null;
  }
  loadSimplifiedData(baseName);
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

const handleDrop = (e) => {
  e.preventDefault();
  const files = e.dataTransfer.files;
  if (files.length > 0 && files[0].type.startsWith("video/")) {
    if (videoUrl.value) {
      URL.revokeObjectURL(videoUrl.value);
    }
    video.value = files[0];
    videoUrl.value = URL.createObjectURL(video.value);
    currentFrame.value = 0;
    currentTime.value = 0;
  }
};
const finalRows = ref([]);
const hasMmData = ref(false);

const finalCsvUrl = computed(() =>
  videoName.value
    ? `/backend/results/elongation_final_${encodeURIComponent(
        videoName.value
      )}.csv`
    : null
);

const pixelToMmCsvUrl = computed(() =>
  videoName.value
    ? `/backend/results/pixel_to_mm_${encodeURIComponent(videoName.value)}.csv`
    : null
);

const simplifiedCsvUrl = computed(() =>
  videoName.value
    ? `/backend/final_data_simplified?base_name=${encodeURIComponent(
        videoName.value
      )}&as_csv=true`
    : null
);
const loadSimplifiedData = async (baseName) => {
  finalRows.value = [];
  hasMmData.value = false;

  try {
    const res = await fetch(
      `/backend/final_data_simplified?base_name=${encodeURIComponent(baseName)}`
    );
    if (!res.ok) return;

    const json = await res.json();
    hasMmData.value = !!json.has_mm;

    finalRows.value = (json.rows || []).map((r) => ({
      timestamp_s: r.timestamp_s != null ? Number(r.timestamp_s) : null,
      elongation_monotonic:
        r.elongation_monotonic != null ? Number(r.elongation_monotonic) : null,
      elongation_monotonic_mm:
        r.elongation_monotonic_mm != null
          ? Number(r.elongation_monotonic_mm)
          : null,
    }));
  } catch (err) {
    console.error("Failed to load simplified data", err);
  }
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
    const res = await fetch("/backend/process", {
      method: "POST",
      body: formData,
    });
    const json = await res.json();

    if (res.ok) {
      logs.value.push("✅ Processing finished.");
      plotUrl.value = `/backend/${json.plot}`;
      loadFirstMarkedImage(json.base_name);
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
    const res = await fetch("/backend/pixel_to_mm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await res.json();
    console.log("Pixel-to-mm result:", result);

    if (res.ok && result.plot) {
      newPlotUrl.value = `/backend/${result.plot}`;
      counter.value++;
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
