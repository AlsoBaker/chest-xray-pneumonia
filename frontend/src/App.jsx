import { useState, useRef, useMemo, useEffect } from "react"
import axios from "axios"
import "./index.css"

const API_URL = "https://vijayesh684-chest-xray-api.hf.space/predict"

// ── Pool of images stored in public/samples/ (all .jpeg) ──
const PNEUMONIA_POOL = [1, 2, 3, 5, 6, 7, 8, 9, 10].map((n) => ({
  label: "PNEUMONIA",
  path: `/samples/pneumonia${n}.jpeg`,
  name: `Pneumonia sample ${n}`,
}))

const NORMAL_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => ({
  label: "NORMAL",
  path: `/samples/normal${n}.jpeg`,
  name: `Normal sample ${n}`,
}))

// Test-set performance from the README (EfficientNetB4 + TTA); the API response overrides these
const DEFAULT_METRICS = {
  accuracy: 93.59,
  roc_auc: 0.9834,
  sensitivity: 94.36,
  specificity: 92.31,
}

// Always pick 2 pneumonia + 2 normal, then shuffle the order
function pickBalanced() {
  const shuffle = (arr) => {
    const a = [...arr]
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[a[i], a[j]] = [a[j], a[i]]
    }
    return a
  }
  return shuffle([...shuffle(PNEUMONIA_POOL).slice(0, 2), ...shuffle(NORMAL_POOL).slice(0, 2)])
}

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches

// Animates a number from 0 to `target` once `target` is set
function useCountUp(target, duration = 1100) {
  const [value, setValue] = useState(0)
  useEffect(() => {
    if (target == null) return
    const ms = prefersReducedMotion() ? 0 : duration
    let raf
    const start = performance.now()
    const tick = (now) => {
      const t = ms ? Math.min(1, (now - start) / ms) : 1
      const eased = 1 - Math.pow(1 - t, 3)
      setValue(target * eased)
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, duration])
  return value
}

export default function App() {
  const [file, setFile]         = useState(null)
  const [preview, setPreview]   = useState(null)
  const [activeSample, setActiveSample] = useState(null)
  const [result, setResult]     = useState(null)
  const [status, setStatus]     = useState("idle") // idle | loading | analyzing
  const [error, setError]       = useState(null)
  const [dragging, setDragging] = useState(false)
  const [split, setSplit]       = useState(50)
  const inputRef                = useRef(null)

  const displayedSamples = useMemo(() => pickBalanced(), [])
  const metrics = result?.metrics ?? DEFAULT_METRICS
  const confidence = useCountUp(result ? Number(result.confidence) : null)
  const isPneumonia = result?.label === "PNEUMONIA"
  const busy = status !== "idle"

  // Release object URLs when the preview changes
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  // ── File handling ──────────────────────────────────────────
  const takeFile = (f, sampleName = null) => {
    if (!f) return
    if (!/\.(png|jpe?g)$/i.test(f.name)) {
      setError("That file isn't a PNG or JPEG. Choose a chest X-ray image in one of those formats.")
      return
    }
    setError(null)
    setResult(null)
    setSplit(50)
    setFile(f)
    setActiveSample(sampleName)
    setPreview(URL.createObjectURL(f))
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    if (!busy) takeFile(e.dataTransfer.files[0])
  }

  const loadSample = async (sample) => {
    setStatus("loading")
    setError(null)
    try {
      const res = await fetch(sample.path)
      if (!res.ok) throw new Error(`the image at ${sample.path} returned ${res.status}`)
      const blob = await res.blob()
      takeFile(new File([blob], sample.name + ".jpg", { type: "image/jpeg" }), sample.name)
    } catch (err) {
      setError(`Couldn't load the sample: ${err.message}.`)
    } finally {
      setStatus("idle")
    }
  }

  // ── API call ───────────────────────────────────────────────
  const analyze = async () => {
    if (!file) return
    setStatus("analyzing")
    setError(null)
    try {
      const formData = new FormData()
      formData.append("file", file)
      const res = await axios.post(API_URL, formData)
      setResult(res.data)
    } catch (err) {
      const code = err.response?.status
      if (code === 503 || code === 502) {
        setError("The model server is asleep or restarting. Wait a minute, then select Analyze X-ray again.")
      } else {
        const reason = err.response?.data?.error || err.message
        setError(`The analysis didn't finish (${reason}). Check your connection and try again.`)
      }
    } finally {
      setStatus("idle")
    }
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setActiveSample(null)
    setError(null)
  }

  const openPicker = () => inputRef.current?.click()

  // ── Lightbox contents ──────────────────────────────────────
  let viewer
  if (result) {
    viewer = (
      <div className="compare" style={{ "--split": `${split}%` }}>
        <img src={result.original} alt="Original chest X-ray" className="compare-img" />
        <img
          src={result.gradcam}
          alt="Grad-CAM heatmap over the X-ray"
          className="compare-img compare-heat"
        />
        <div className="compare-handle" aria-hidden="true" />
        <input
          type="range"
          className="compare-range"
          min="0"
          max="100"
          value={split}
          onChange={(e) => setSplit(Number(e.target.value))}
          aria-label="Drag to compare the original X-ray with the heatmap"
        />
        <span className="compare-tag compare-tag-left">Original</span>
        <span className="compare-tag compare-tag-right">Heatmap</span>
      </div>
    )
  } else if (preview) {
    viewer = (
      <div className={`film ${status === "analyzing" ? "is-scanning" : ""}`}>
        <img src={preview} alt={activeSample ?? "Your uploaded X-ray"} />
        <div className="scanline" aria-hidden="true" />
      </div>
    )
  } else {
    viewer = (
      <div className="empty">
        <svg className="empty-icon" viewBox="0 0 64 64" aria-hidden="true">
          <path d="M32 8v20M32 28c-4 6-14 4-18 14-3 8 0 14 6 14s8-6 8-14M32 28c4 6 14 4 18 14 3 8 0 14-6 14s-8-6-8-14" />
        </svg>
        <p className="empty-title">Drop a chest X-ray here</p>
        <p className="empty-sub">PNG or JPEG, frontal view</p>
        <button type="button" className="btn btn-dark" onClick={openPicker}>
          Choose an X-ray
        </button>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">
          <span className="brand-mark" aria-hidden="true" />
          Pneumonia X-ray reader
        </span>
        <a
          className="topbar-link"
          href="https://github.com/AlsoBaker/chest-xray-pneumonia-detection"
          target="_blank"
          rel="noreferrer"
        >
          Source on GitHub
        </a>
      </header>

      <main className="stage">
        {/* ── Lightbox (hero) ── */}
        <section
          className={`lightbox ${dragging ? "is-dragging" : ""} ${result ? "has-result" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          aria-label="X-ray viewer"
        >
          <div className="lightbox-panel">{viewer}</div>
          {result && (
            <div className="legend">
              <span>Less attention</span>
              <span className="legend-ramp" aria-hidden="true" />
              <span>More attention</span>
            </div>
          )}
          {status === "analyzing" && (
            <p className="lightbox-caption">
              Reading the X-ray… The first run can take up to a minute while the server wakes up.
            </p>
          )}
        </section>

        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg"
          hidden
          onChange={(e) => { takeFile(e.target.files[0]); e.target.value = "" }}
        />

        {/* ── Control column ── */}
        <section className="panel">
          {!result ? (
            <div className="intro" key="intro">
              <h1 className="headline">See where the model looks.</h1>
              <p className="lede">
                Upload a frontal chest X-ray. An EfficientNetB4 network trained on 5,863 scans says
                whether it shows signs of pneumonia, and a Grad-CAM heatmap shows which regions
                drove that call.
              </p>

              <div className="actions">
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={analyze}
                  disabled={!file || busy}
                >
                  {status === "analyzing" ? "Analyzing…" : "Analyze X-ray"}
                </button>
                <button type="button" className="btn btn-ghost" onClick={openPicker} disabled={busy}>
                  {file ? "Choose a different image" : "Upload your own"}
                </button>
              </div>

              <div className="samples">
                <p className="samples-title">Or try a sample from the test set</p>
                <ul className="sample-grid">
                  {displayedSamples.map((sample) => (
                    <li key={sample.path}>
                      <button
                        type="button"
                        className={`sample ${sample.label.toLowerCase()} ${
                          activeSample === sample.name ? "is-active" : ""
                        }`}
                        onClick={() => loadSample(sample)}
                        disabled={busy}
                        aria-pressed={activeSample === sample.name}
                      >
                        <img src={sample.path} alt="" width="160" height="160" />
                        <span className="sample-tag">
                          {sample.label === "PNEUMONIA" ? "Pneumonia" : "Normal"}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <div className={`verdict ${isPneumonia ? "pneumonia" : "normal"}`} key="verdict">
              <h1 className="verdict-label">
                {isPneumonia ? "Signs of pneumonia" : "No signs of pneumonia"}
              </h1>

              <div className="meter">
                <div className="meter-head">
                  <span>Model confidence</span>
                  <span className="meter-value">{confidence.toFixed(1)}%</span>
                </div>
                <div className="meter-track">
                  <div className="meter-fill" style={{ width: `${result.confidence}%` }} />
                </div>
              </div>

              <p className="verdict-detail">
                The model scored this image <strong>{result.score}</strong>. Anything above{" "}
                <strong>{result.threshold}</strong> is read as pneumonia.
                {activeSample && (
                  <> This sample is labelled <strong>{activeSample.startsWith("Pneumonia") ? "pneumonia" : "normal"}</strong> in the dataset.</>
                )}
              </p>

              <p className="verdict-hint">
                Drag across the image to compare the original with the heatmap.
              </p>

              <button type="button" className="btn btn-primary" onClick={reset}>
                Analyze another X-ray
              </button>
            </div>
          )}

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </section>
      </main>

      {/* Persistent live region so screen readers hear progress and results */}
      <p className="sr-only" aria-live="polite">
        {status === "analyzing"
          ? "Analyzing the X-ray."
          : result
            ? `${isPneumonia ? "Signs of pneumonia" : "No signs of pneumonia"}, ${result.confidence}% confidence.`
            : ""}
      </p>

      <footer className="footnote">
        <dl className="stats" aria-label="Model performance on the held-out test set">
          <div><dt>Accuracy</dt><dd>{metrics.accuracy}%</dd></div>
          <div><dt>Sensitivity</dt><dd>{metrics.sensitivity}%</dd></div>
          <div><dt>Specificity</dt><dd>{metrics.specificity}%</dd></div>
          <div><dt>ROC-AUC</dt><dd>{metrics.roc_auc}</dd></div>
        </dl>
        <p className="disclaimer">
          Measured on the held-out test set. This is a research project, not a medical device.
          Talk to a clinician about any X-ray that matters.
        </p>
      </footer>
    </div>
  )
}
