<template>
  <div class="bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-6">
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-lg font-semibold text-white">Processed Videos</h2>
      <button
        class="flex items-center px-3 py-2 text-sm font-medium text-gray-300 bg-gray-700 rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-800"
        :disabled="loading"
        @click="fetchProcessedVideos"
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
            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
          />
        </svg>
        {{ loading ? "Loading..." : "Refresh" }}
      </button>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex items-center justify-center py-12">
      <div class="flex items-center">
        <svg
          class="w-6 h-6 text-blue-400 animate-spin mr-3"
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
        <span class="text-gray-400">Loading processed videos...</span>
      </div>
    </div>

    <!-- Error State -->
    <div
      v-else-if="error"
      class="bg-red-900 border border-red-700 rounded-md p-4"
    >
      <div class="flex items-center">
        <svg
          class="w-5 h-5 text-red-400 mr-2"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fill-rule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clip-rule="evenodd"
          />
        </svg>
        <span class="text-red-300">{{ error }}</span>
      </div>
    </div>

    <!-- Empty State -->
    <div v-else-if="processedVideos.length === 0" class="text-center py-12">
      <svg
        class="mx-auto h-12 w-12 text-gray-500"
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
      <h3 class="mt-2 text-sm font-medium text-white">No processed videos</h3>
      <p class="mt-1 text-sm text-gray-400">
        Get started by uploading and processing a video.
      </p>
    </div>

    <!-- Videos List -->
    <div v-else class="space-y-3">
      <div
        v-for="video in processedVideos"
        :key="video"
        class="group relative bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-colors border border-gray-600"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center flex-1 min-w-0">
            <div class="flex-shrink-0">
              <svg
                class="w-8 h-8 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div class="ml-3 flex-1 min-w-0">
              <p class="text-sm font-medium text-white truncate">{{ video }}</p>
              <p class="text-xs text-gray-400">
                Processed video data available
              </p>
            </div>
          </div>

          <div class="flex items-center space-x-2 ml-4">
            <button
              class="flex items-center px-3 py-1.5 text-sm font-medium text-blue-400 bg-blue-900 rounded-md hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-700"
              @click="$emit('select', video)"
            >
              <svg
                class="w-4 h-4 mr-1"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
              View
            </button>

            <button
              class="flex items-center px-3 py-1.5 text-sm font-medium text-red-400 bg-red-900 rounded-md hover:bg-red-800 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-700 opacity-0 group-hover:opacity-100 transition-opacity"
              @click="confirmDelete(video)"
            >
              <svg
                class="w-4 h-4 mr-1"
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
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Delete Confirmation Modal -->
    <div
      v-if="showDeleteModal"
      class="fixed inset-0 bg-black bg-opacity-50 overflow-y-auto h-full w-full z-50"
    >
      <div
        class="relative top-20 mx-auto p-5 border border-gray-600 w-96 shadow-xl rounded-md bg-gray-800"
      >
        <div class="mt-3 text-center">
          <div
            class="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-900"
          >
            <svg
              class="h-6 w-6 text-red-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
          </div>
          <h3 class="text-lg font-medium text-white mt-4">Delete Video Data</h3>
          <div class="mt-2 px-7 py-3">
            <p class="text-sm text-gray-400">
              Are you sure you want to delete all data for
              <span class="font-medium text-white">"{{ videoToDelete }}"</span>?
              This action cannot be undone.
            </p>
          </div>
          <div class="flex justify-center space-x-3 mt-4">
            <button
              class="px-4 py-2 bg-gray-600 text-gray-300 rounded-md hover:bg-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-500"
              @click="showDeleteModal = false"
            >
              Cancel
            </button>
            <button
              class="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500"
              @click="deleteVideo"
            >
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";

const processedVideos = ref([]);
const loading = ref(false);
const error = ref(null);
const selectedVideo = ref(null);
const showDeleteModal = ref(false);
const videoToDelete = ref(null);

const emit = defineEmits(["select", "delete"]);

const fetchProcessedVideos = async () => {
  loading.value = true;
  error.value = null;
  try {
    const res = await fetch("/backend/list_processed/");
    if (!res.ok) throw new Error("Failed to fetch processed videos");
    const json = await res.json();
    processedVideos.value = json.videos;
  } catch (err) {
    error.value = err.message;
  } finally {
    loading.value = false;
  }
};

const confirmDelete = (videoName) => {
  videoToDelete.value = videoName;
  showDeleteModal.value = true;
};

const deleteVideo = async () => {
  if (!videoToDelete.value) return;

  try {
    const res = await fetch(
      `/backend/delete_processed/?video_name=${encodeURIComponent(
        videoToDelete.value
      )}`,
      {
        method: "DELETE",
      }
    );
    if (!res.ok) throw new Error("Failed to delete video data");

    processedVideos.value = processedVideos.value.filter(
      (v) => v !== videoToDelete.value
    );
    if (selectedVideo.value === videoToDelete.value) selectedVideo.value = null;
    emit("delete", videoToDelete.value);

    showDeleteModal.value = false;
    videoToDelete.value = null;
  } catch (err) {
    alert(`Error deleting video: ${err.message}`);
  }
};

onMounted(() => {
  fetchProcessedVideos();
});
</script>
