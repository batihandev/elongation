<script setup>
import { ref, onMounted } from "vue";

const processedVideos = ref([]);
const loading = ref(false);
const error = ref(null);
const selectedVideo = ref(null);
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

const deleteVideo = async (videoName) => {
  if (!confirm(`Delete all data for "${videoName}"?`)) return;
  try {
    const res = await fetch(
      `/backend/delete_processed/?video_name=${encodeURIComponent(videoName)}`,
      {
        method: "DELETE",
      }
    );
    if (!res.ok) throw new Error("Failed to delete video data");
    processedVideos.value = processedVideos.value.filter(
      (v) => v !== videoName
    );
    if (selectedVideo.value === videoName) selectedVideo.value = null;
    emit("delete", videoName);
  } catch (err) {
    alert(`Error deleting video: ${err.message}`);
  }
};

onMounted(() => {
  fetchProcessedVideos();
});
</script>

<template>
  <div>
    <h2 class="text-xl font-bold mb-4">Processed Videos</h2>
    <div v-if="loading">Loading...</div>
    <div v-if="error" class="text-red-600">{{ error }}</div>
    <div v-if="processedVideos.length === 0">No processed videos found.</div>
    <div class="space-y-2">
      <div
        v-for="video in processedVideos"
        :key="video"
        class="border p-4 rounded flex justify-between items-center cursor-pointer hover:bg-blue-400"
      >
        <div class="flex-grow" @click="$emit('select', video)">{{ video }}</div>
        <button
          class="bg-red-600 text-white px-3 py-1 rounded"
          @click.stop="deleteVideo(video)"
        >
          Delete
        </button>
      </div>
    </div>
  </div>
</template>
