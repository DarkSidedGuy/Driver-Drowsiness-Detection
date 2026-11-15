
const $ = (id) => document.getElementById(id);

let chart = null;
let currentSessionId = null;
let lastError = "";


// ============================================================
// 1. GENERAL HELPERS
// ============================================================

function toast(message) {

    const el = $("toast");

    el.textContent = message;

    el.classList.add("show");

    setTimeout(() => {
        el.classList.remove("show");
    }, 2800);
}


function escapeHTML(value) {

    return String(value ?? "").replace(
        /[&<>"']/g,
        c => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        }[c])
    );
}


function formatDuration(seconds) {

    const n = Math.max(
        0,
        Math.floor(Number(seconds) || 0)
    );

    const h = Math.floor(n / 3600);

    const m = Math.floor((n % 3600) / 60);

    const s = n % 60;

    return h
        ? `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`
        : `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
}


async function getJSON(url) {

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
    }

    return response.json();
}


// ============================================================
// 2. CAMERA DISPLAY
// ============================================================

function setCamera(active) {

    const img = $("cameraFeed");

    const placeholder = $("cameraPlaceholder");

    if (active) {

        if (
            !img.src ||
            !img.src.includes("/video_feed")
        ) {
            img.src = "/video_feed";
        }

        img.style.display = "block";

        placeholder.style.display = "none";

    } else {

        img.removeAttribute("src");

        img.style.display = "none";

        placeholder.style.display = "flex";
    }
}


// ============================================================
// 3. STATUS DISPLAY
// ============================================================

function setStatus(data) {

    const status = data.status || "Idle";

    $("statusTitle").textContent = status;

    $("statusMessage").textContent =
        data.message || "";

    const pill = $("statusPill");

    pill.textContent = status.toUpperCase();

    pill.className = "status-pill " + (

        status === "Monitoring"
            ? "active"

            : status === "Drowsiness Alert"
                ? "warning"

                : status === "Error"
                    ? "error"

                    : "idle"
    );

    $("statusBanner").style.borderColor =
        status === "Drowsiness Alert"
            ? "#713448"
            : "";

    $("startBtn").disabled = !!data.running;

    $("stopBtn").disabled = !data.running;

    $("cameraOverlay").textContent =
        data.running
            ? (
                data.face_detected
                    ? "FACE TRACKING ACTIVE"
                    : "SEARCHING FOR FACE"
            )
            : "WAITING FOR CAMERA";

    setCamera(!!data.running);


    // Face detection state

    $("faceState").textContent =
        data.face_detected
            ? "FACE DETECTED"
            : "NO FACE";

    $("faceState").classList.toggle(
        "detected",
        !!data.face_detected
    );


    // Eye state

    $("eyeState").textContent =
        data.ear > 0
            ? (
                data.ear < 0.23
                    ? "Closed / low"
                    : "Open"
            )
            : "Waiting";


    // Yawn state

    $("yawnState").textContent =
        data.mar > 0.55
            ? "Mouth open"
            : (
                data.face_detected
                    ? "Normal"
                    : "Waiting"
            );


    // Alert box

    const alertBox = $("alertBox");

    const warning =
        status === "Drowsiness Alert";

    alertBox.classList.toggle(
        "warning",
        warning
    );

    alertBox.querySelector("b").textContent =
        warning
            ? "Drowsiness warning"
            : "Safety status";

    alertBox.querySelector("span").textContent =
        warning
            ? "Prolonged eye closure has been detected. The prototype alert has been triggered."
            : "No active drowsiness alert. This prototype is for educational demonstration only.";
}


// ============================================================
// 4. LIVE STATUS AND METRICS
// ============================================================

async function refreshStatus() {

    try {

        const data = await getJSON("/api/status");

        currentSessionId =
            data.session_id || currentSessionId;


        // Main metric cards

        $("earValue").textContent =
            Number(data.ear || 0).toFixed(3);

        $("marValue").textContent =
            Number(data.mar || 0).toFixed(3);

        $("perclosValue").innerHTML =
            `${Number(data.perclos || 0).toFixed(1)}<small>%</small>`;

        $("durationValue").textContent =
            formatDuration(data.session_elapsed);


        // Detection counts

        $("drowsyCount").textContent =
            data.drowsiness_count ?? 0;

        $("yawnCount").textContent =
            data.yawn_count ?? 0;


        // NEW: Blink analytics

        $("currentClosure").textContent =
            `${Number(data.current_eye_closure || 0).toFixed(2)}s`;

        $("blinkRate").textContent =
            `${Number(data.blink_rate || 0).toFixed(0)}/min`;

        $("avgBlink").textContent =
            `${Number(data.avg_blink_duration || 0).toFixed(2)}s`;

        $("longestClosure").textContent =
            `${Number(data.longest_eye_closure || 0).toFixed(2)} sec`;


        // Session analytics

        $("maxPerclos").textContent =
            `${Number(data.max_perclos || 0).toFixed(1)}%`;

        $("validTime").textContent =
            `${Number(data.valid_monitoring_time || 0).toFixed(0)} sec`;

        $("drowsyDuration").textContent =
            `${Number(data.drowsy_duration || 0).toFixed(1)} sec`;


        // Update status

        setStatus(data);


        // Show errors only once

        if (
            data.error &&
            data.error !== lastError
        ) {

            lastError = data.error;

            toast(data.error);
        }


        await refreshSeries();

    } catch (err) {

        console.error("Status update error:", err);
    }
}


// ============================================================
// 5. EVENT HISTORY
// ============================================================

async function refreshEvents() {

    try {

        const events = await getJSON("/api/events");

        $("eventCount").textContent =
            events.length;

        $("eventList").innerHTML =

            events.length

                ? events.map(e => {

                    const type = String(
                        e.type || ""
                    ).toLowerCase();

                    return `

                        <div class="event-item">

                            <i class="event-dot ${escapeHTML(type)}"></i>

                            <div class="event-text">

                                <b>${escapeHTML(e.type)}</b>

                                <span>${escapeHTML(e.message)}</span>

                            </div>

                            <span class="event-time">
                                ${escapeHTML(e.time || "")}
                            </span>

                        </div>

                    `;

                }).join("")

                : `
                    <div class="empty-state">
                        No events recorded yet.
                    </div>
                `;

    } catch (err) {

        console.error("Event update error:", err);
    }
}


// ============================================================
// 6. LIVE ANALYTICS GRAPH
// ============================================================

async function refreshSeries() {

    try {

        const rows = await getJSON("/api/series");

        if (!chart) return;

        chart.data.labels = rows.map(
            r => `${r.t}s`
        );

        chart.data.datasets[0].data =
            rows.map(r => r.ear);

        chart.data.datasets[1].data =
            rows.map(r => r.mar);

        chart.data.datasets[2].data =
            rows.map(r => r.perclos);

        chart.update("none");

    } catch (err) {

        console.error("Chart update error:", err);
    }
}


// ============================================================
// 7. SESSION HISTORY
// ============================================================

async function refreshHistory() {

    try {

        const rows = await getJSON("/api/history");

        $("historyBody").innerHTML =

            rows.length

                ? rows.map(r => {

                    const sid = encodeURIComponent(
                        r.session_id
                    );

                    return `

                        <tr>

                            <td>${escapeHTML(r.date)}</td>

                            <td>${escapeHTML(r.drowsiness_events)}</td>

                            <td>${escapeHTML(r.yawns)}</td>

                            <td>
                                ${escapeHTML(r.max_perclos_percent)}%
                            </td>

                            <td>
                                <a href="/api/report/${sid}" target="_blank">
                                    PDF ↗
                                </a>
                            </td>

                        </tr>

                    `;

                }).join("")

                : `
                    <tr>
                        <td colspan="5" class="empty-cell">
                            No saved sessions yet.
                        </td>
                    </tr>
                `;

    } catch (err) {

        console.error("History update error:", err);
    }
}


// ============================================================
// 8. CHART INITIALIZATION
// ============================================================

function initializeChart() {

    const ctx = $("trendChart").getContext("2d");

    chart = new Chart(ctx, {

        type: "line",

        data: {

            labels: [],

            datasets: [

                {
                    label: "EAR",
                    data: [],
                    borderColor: "#4dd8e8",
                    backgroundColor: "#4dd8e822",
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: "y"
                },

                {
                    label: "MAR",
                    data: [],
                    borderColor: "#a78bfa",
                    backgroundColor: "#a78bfa22",
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: "y"
                },

                {
                    label: "PERCLOS",
                    data: [],
                    borderColor: "#f7bd55",
                    backgroundColor: "#f7bd5522",
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: "y1"
                }

            ]
        },


        options: {

            responsive: true,

            maintainAspectRatio: false,

            animation: false,

            interaction: {
                mode: "index",
                intersect: false
            },

            plugins: {

                legend: {
                    display: false
                },

                tooltip: {

                    backgroundColor: "#0b1424",

                    borderColor: "#30435d",

                    borderWidth: 1,

                    titleColor: "#e8eef8",

                    bodyColor: "#b8c7da"
                }
            },


            scales: {

                x: {

                    grid: {
                        color: "#1d2a3d"
                    },

                    ticks: {
                        color: "#71839b",
                        maxTicksLimit: 9,
                        font: {
                            size: 9
                        }
                    }
                },


                y: {

                    position: "left",

                    min: 0,

                    max: 1,

                    grid: {
                        color: "#1d2a3d"
                    },

                    ticks: {
                        color: "#71839b",
                        font: {
                            size: 9
                        }
                    }
                },


                y1: {

                    position: "right",

                    min: 0,

                    max: 100,

                    grid: {
                        drawOnChartArea: false
                    },

                    ticks: {
                        color: "#71839b",
                        font: {
                            size: 9
                        }
                    },

                    title: {

                        display: true,

                        text: "PERCLOS %",

                        color: "#71839b"
                    }
                }
            }
        }
    });
}


// ============================================================
// 9. START AND STOP MONITORING
// ============================================================

async function startMonitoring() {

    try {

        const response = await fetch(
            "/api/start",
            {
                method: "POST"
            }
        );

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.message || "Could not start monitoring"
            );
        }

        toast(
            data.message || "Monitoring started"
        );

        lastError = "";

    } catch (err) {

        toast(err.message);
    }
}


async function stopMonitoring() {

    try {

        const response = await fetch(
            "/api/stop",
            {
                method: "POST"
            }
        );

        const data = await response.json();

        toast(
            data.message || "Stopping monitoring"
        );

        // Allow the background worker to save the summary.
        setTimeout(refreshHistory, 1800);

        setTimeout(refreshEvents, 700);

        setTimeout(refreshStatus, 700);

    } catch (err) {

        toast(err.message);
    }
}


// ============================================================
// 10. BUTTON EVENTS
// ============================================================

$("startBtn").addEventListener(
    "click",
    startMonitoring
);

$("stopBtn").addEventListener(
    "click",
    stopMonitoring
);

$("refreshHistory").addEventListener(
    "click",
    refreshHistory
);


$("downloadSummary").addEventListener(
    "click",
    () => {
        window.location.href =
            "/api/download/summary";
    }
);


$("downloadEvents").addEventListener(
    "click",
    () => {

        if (!currentSessionId) {
            return toast("Start a session first.");
        }

        window.location.href =
            `/api/download/events?session_id=${encodeURIComponent(currentSessionId)}`;
    }
);


$("downloadPdf").addEventListener(
    "click",
    () => {

        if (!currentSessionId) {
            return toast("Start a session first.");
        }

        window.location.href =
            `/api/report/${encodeURIComponent(currentSessionId)}`;
    }
);


// ============================================================
// 11. INITIALIZATION
// ============================================================

$("todayLabel").textContent =
    new Date().toLocaleDateString(
        undefined,
        {
            weekday: "short",
            month: "short",
            day: "numeric",
            year: "numeric"
        }
    );


initializeChart();

setCamera(false);

refreshStatus();

refreshEvents();

refreshHistory();


// Dashboard refresh intervals

setInterval(refreshStatus, 1000);

setInterval(refreshEvents, 2500);
