// FleetSense Pure Prediction Client

function getApiBaseUrl() {
  if (window.__FLEETSENSE_API_URL__) {
    return window.__FLEETSENSE_API_URL__.replace(/\/$/, "");
  }
  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    return "http://localhost:8000";
  }
  // In Azure Container Apps: fleetsense-frontend.xxx -> fleetsense-api.xxx
  const currentHost = window.location.host;
  if (currentHost.includes("fleetsense-frontend")) {
    const apiHost = currentHost.replace("fleetsense-frontend", "fleetsense-api");
    return `${window.location.protocol}//${apiHost}`;
  }
  // In generic environments (IP address / custom domain), target API port 8000 on the same hostname
  if (window.location.protocol && window.location.hostname) {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return "";
}

const apiBaseUrl = getApiBaseUrl();

// Navigation Tabs
const tabBtnSingle = document.getElementById("tab-btn-single");
const tabBtnBatch = document.getElementById("tab-btn-batch");
const sectionSingle = document.getElementById("section-single");
const sectionBatch = document.getElementById("section-batch");

tabBtnSingle.addEventListener("click", () => {
  tabBtnSingle.className = "px-3.5 py-1.5 rounded-lg font-medium transition bg-indigo-600 text-white shadow";
  tabBtnBatch.className = "px-3.5 py-1.5 rounded-lg font-medium transition text-slate-400 hover:text-white";
  sectionSingle.classList.remove("hidden");
  sectionBatch.classList.add("hidden");
});

tabBtnBatch.addEventListener("click", () => {
  tabBtnBatch.className = "px-3.5 py-1.5 rounded-lg font-medium transition bg-indigo-600 text-white shadow";
  tabBtnSingle.className = "px-3.5 py-1.5 rounded-lg font-medium transition text-slate-400 hover:text-white";
  sectionBatch.classList.remove("hidden");
  sectionSingle.classList.add("hidden");
});

// Single Prediction
const singleForm = document.getElementById("single-form");
const btnFillSample = document.getElementById("btn-fill-sample");
const btnPredictSingle = document.getElementById("btn-predict-single");
const singlePlaceholder = document.getElementById("single-placeholder");
const singleLoading = document.getElementById("single-loading");
const singleResult = document.getElementById("single-result");
const singleError = document.getElementById("single-error");
const singleErrorText = document.getElementById("single-error-text");
const singleMinutes = document.getElementById("single-minutes");
const resDistance = document.getElementById("res-distance");
const resRoute = document.getElementById("res-route");

btnFillSample.addEventListener("click", () => {
  document.getElementById("PULocationID").value = 161; // Midtown
  document.getElementById("DOLocationID").value = 237; // Upper East Side
  document.getElementById("trip_distance").value = 3.2;
  document.getElementById("passenger_count").value = 1;
  document.getElementById("pickup_hour").value = 17;
  document.getElementById("pickup_dayofweek").value = 4;
});

function computePickupDatetime(pickupHour, pickupDayOfWeek) {
  // Compute next or recent date matching target day of week (0=Mon, ..., 6=Sun)
  const d = new Date();
  const currentDay = (d.getDay() + 6) % 7; // Convert JS 0=Sun to 0=Mon
  let diff = pickupDayOfWeek - currentDay;
  d.setDate(d.getDate() + diff);
  d.setHours(pickupHour, 0, 0, 0);
  return d.toISOString();
}

singleForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const pu = parseInt(document.getElementById("PULocationID").value, 10);
  const do_loc = parseInt(document.getElementById("DOLocationID").value, 10);
  const dist = parseFloat(document.getElementById("trip_distance").value);
  const hour = parseInt(document.getElementById("pickup_hour").value, 10);
  const day = parseInt(document.getElementById("pickup_dayofweek").value, 10);

  const payload = {
    VendorID: 1,
    pickup_datetime: computePickupDatetime(hour, day),
    PULocationID: pu,
    DOLocationID: do_loc,
    trip_distance: dist,
    passenger_count: parseInt(document.getElementById("passenger_count").value, 10)
  };

  singlePlaceholder.classList.add("hidden");
  singleResult.classList.add("hidden");
  singleError.classList.add("hidden");
  singleLoading.classList.remove("hidden");
  btnPredictSingle.disabled = true;

  try {
    const res = await fetch(`${apiBaseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `Error ${res.status}` }));
      let msg = err.detail;
      if (typeof msg === "object") {
        msg = JSON.stringify(msg);
      }
      throw new Error(msg || "Prediction request failed.");
    }

    const data = await res.json();
    const duration = data.predicted_duration_minutes != null 
      ? data.predicted_duration_minutes 
      : (data.prediction_minutes != null ? data.prediction_minutes : null);

    singleMinutes.textContent = duration != null ? duration.toFixed(1) : "--";
    resDistance.textContent = `${dist.toFixed(1)} miles`;
    resRoute.textContent = `Zone ${pu} to Zone ${do_loc}`;

    singleLoading.classList.add("hidden");
    singleResult.classList.remove("hidden");
  } catch (err) {
    singleLoading.classList.add("hidden");
    singleError.classList.remove("hidden");
    singleErrorText.textContent = typeof err.message === "object" ? JSON.stringify(err.message) : err.message;
  } finally {
    btnPredictSingle.disabled = false;
  }
});

// Batch Prediction
const batchInput = document.getElementById("batch-input");
const btnLoadSampleBatch = document.getElementById("btn-load-sample-batch");
const btnPredictBatch = document.getElementById("btn-predict-batch");
const batchLoading = document.getElementById("batch-loading");
const batchError = document.getElementById("batch-error");
const batchErrorText = document.getElementById("batch-error-text");
const batchResult = document.getElementById("batch-result");
const batchTableBody = document.getElementById("batch-table-body");
const batchCount = document.getElementById("batch-count");

const nowIso = new Date().toISOString();
const sampleBatchData = [
  { "VendorID": 1, "pickup_datetime": nowIso, "PULocationID": 161, "DOLocationID": 237, "trip_distance": 2.5, "passenger_count": 1 },
  { "VendorID": 1, "pickup_datetime": nowIso, "PULocationID": 79, "DOLocationID": 142, "trip_distance": 4.8, "passenger_count": 2 },
  { "VendorID": 1, "pickup_datetime": nowIso, "PULocationID": 138, "DOLocationID": 230, "trip_distance": 9.2, "passenger_count": 1 }
];

btnLoadSampleBatch.addEventListener("click", () => {
  batchInput.value = JSON.stringify(sampleBatchData, null, 2);
});

// Load sample batch on startup
batchInput.value = JSON.stringify(sampleBatchData, null, 2);

btnPredictBatch.addEventListener("click", async () => {
  let parsedPayload;
  try {
    parsedPayload = JSON.parse(batchInput.value);
    if (!Array.isArray(parsedPayload)) {
      throw new Error("Batch data must be a JSON array of trips.");
    }
  } catch (err) {
    batchError.classList.remove("hidden");
    batchErrorText.textContent = "Invalid JSON: " + err.message;
    return;
  }

  batchError.classList.add("hidden");
  batchResult.classList.add("hidden");
  batchLoading.classList.remove("hidden");
  btnPredictBatch.disabled = true;

  try {
    const res = await fetch(`${apiBaseUrl}/predict/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trips: parsedPayload }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `Error ${res.status}` }));
      let msg = err.detail;
      if (typeof msg === "object") {
        msg = JSON.stringify(msg);
      }
      throw new Error(msg || "Batch prediction request failed.");
    }

    const data = await res.json();
    const predictions = data.predictions || [];

    batchTableBody.innerHTML = "";
    predictions.forEach((item, idx) => {
      const original = parsedPayload[idx] || {};
      const duration = item.predicted_duration_minutes != null 
        ? item.predicted_duration_minutes 
        : (item.prediction_minutes != null ? item.prediction_minutes : null);

      const row = document.createElement("tr");
      row.className = "hover:bg-slate-800/50";
      row.innerHTML = `
        <td class="px-4 py-2 font-mono text-slate-400">${idx + 1}</td>
        <td class="px-4 py-2 font-mono">${original.PULocationID ?? "--"}</td>
        <td class="px-4 py-2 font-mono">${original.DOLocationID ?? "--"}</td>
        <td class="px-4 py-2 font-mono">${original.trip_distance ?? "--"}</td>
        <td class="px-4 py-2 font-bold text-indigo-400">${duration != null ? duration.toFixed(1) + " min" : "--"}</td>
      `;
      batchTableBody.appendChild(row);
    });

    batchCount.textContent = predictions.length;
    batchLoading.classList.add("hidden");
    batchResult.classList.remove("hidden");
  } catch (err) {
    batchLoading.classList.add("hidden");
    batchError.classList.remove("hidden");
    batchErrorText.textContent = typeof err.message === "object" ? JSON.stringify(err.message) : err.message;
  } finally {
    btnPredictBatch.disabled = false;
  }
});
