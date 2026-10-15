# Manga2Motion: AI-Powered Manga-to-Video Diffusion System
## Architecture Specification, Timeline Scheduling & Implementation Plan

---

## 1. Executive Summary & Objective

**Manga2Motion** is an end-to-end generative software pipeline designed to convert multi-page static Manga scenes into cohesive, colorized, and temporally consistent motion video.

### Target Deliverable & Minimum Requirement
* **Input**: ~3 consecutive pages of standard black-and-white / screentoned manga containing an action or narrative scene.
* **Output**: A synchronized **30-frame video sequence** (standard baseline: 1.25s @ 24fps or 1.0s @ 30fps, expandable to multi-beat shots) with:
  1. Automated panel segmentation and narrative reading order determination.
  2. Text bubble & SFX (sound effects) removal with structural background inpainting.
  3. Style-coherent character and background colorization with palette reference locking.
  4. Camera and character motion generation driven by Image-to-Video (I2V) diffusion.
  5. Temporal frame interpolation (RIFE / FILM) and deflickering polish.
* **Interfaces**: Python CLI pipeline + Interactive WebUI (Gradio/FastAPI) featuring a timeline editor, panel selector, and color reference palette manager.

```
+--------------------------------------------------------------------------------------------------+
|                                    MANGA2MOTION HIGH-LEVEL FLOW                                  |
|                                                                                                  |
|  [3 Manga Pages]                                                                                 |
|         │                                                                                        |
|         ▼                                                                                        |
|  ┌───────────────┐       ┌────────────────┐       ┌─────────────────┐       ┌─────────────────┐  |
|  │ 1. Ingestion  │ ────> │ 2. Bubble &    │ ────> │ 3. Lineart &    │ ────> │ 4. Reference    │  |
|  │    & Panel    │       │    SFX Removal │       │    Layer        │       │    Colorization │  |
|  │    Ordering   │       │    (Inpainting)│       │    Decomp (SAM2)│       │    (SDXL/IP-Ad.)│  |
|  └───────────────┘       └────────────────┘       └─────────────────┘       └─────────────────┘  |
|                                                                                      │           |
|         ┌────────────────────────────────────────────────────────────────────────────┘           |
|         ▼                                                                                        |
|  ┌───────────────┐       ┌────────────────┐       ┌─────────────────┐       ┌─────────────────┐  |
|  │ 5. Timeline   │ ────> │ 6. I2V Motion  │ ────> │ 7. Frame Interp │ ────> │ 8. 30-Frame     │  |
|  │    Shot Setup │       │    Diffusion   │       │    & Deflicker  │       │    Video Render │  |
|  │    (Keyframes)│       │    (AnimateDiff│       │    (RIFE / FILM)│       │    (MP4/ProRes) │  |
|  └───────────────┘       └────────────────┘       └─────────────────┘       └─────────────────┘  |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Technical Challenges & Architectural Solutions

| Challenge | Root Cause in Manga | Manga2Motion Solution |
| :--- | :--- | :--- |
| **Screentone Artifacts** | Halftone dots and hatching interfere with diffusion denoisers | Bilateral tone-smoothing + Manga Lineart Extraction (ControlNet Lineart Anime preprocessor) |
| **Text Bubbles & SFX** | Onomatopoeia and dialog cover character anatomy and backgrounds | Manga-OCR / YOLO bubble detector + Fast High-Res Inpainting (LaMa / ProPainter) |
| **Color Drift across Panels** | Individual diffusion colorization yields mismatched hair, eye, and skin tones | Palette Memory System: IP-Adapter anime reference + Cross-Attention Control + Color Histogram Matching |
| **Motion Incoherence** | Standard I2V diffusion hallucinates shapes without structural anchors | Hybrid Motion Guidance: SAM2 foreground/background depth splitting (Depth-Anything-V2) + Motion ControlNet |
| **Frame Budget Allocation** | Fitting a 3-page narrative into 30 frames without abrupt cuts | Timeline Scene Director: Dynamic shot budget allocator (e.g. 10 frames setup -> 12 frames climax action -> 8 frames resolution) + Interpolation |

---

## 3. End-to-End Pipeline Breakdown

### Phase 1: Manga Ingestion, Panel Extraction & Narrative Graph
* **Panel Detection**: Fine-tuned YOLOv8-Manga model outputs bounding boxes for all panels on each page.
* **Reading Order Traversal**: Implements right-to-left, top-to-bottom Japanese manga reading order graph traversal:
  $$\text{Sort Priority} = (Y_{\text{row}}, -X_{\text{col}})$$
  with support for non-rectangular and bleeding panels.
* **Text Bubble & SFX Inpainting**:
  * Detect dialog bubbles via contour and text mask segmentation.
  * Inpaint hidden areas using **LaMa (Large Mask Inpainting)** to produce clean background art and complete character contours.

### Phase 2: Layer Decomposition & Depth Estimation
* **Foreground/Background Separation**:
  * Segment characters, props, and background using **SAM 2 (Segment Anything 2)**.
  * Inpaint occluded background plates to allow multi-plane 2.5D camera movements.
* **Depth Map Extraction**:
  * Run **Depth-Anything-V2 (Anime finetune)** or **Marigold** to produce dense depth maps for camera panning and ControlNet depth conditioning.

### Phase 3: Reference-Consistent Colorization Engine
* **Color Reference Palette**:
  * Users can upload a colored character reference sheet or specify HEX palettes for key entities (e.g., `#FF4500` hair, `#2B4C7E` coat).
* **Diffusion Colorization**:
  * Model: **SDXL / Animagine XL / NovelAI** architecture.
  * Conditioners:
    1. **ControlNet Lineart Anime**: Enforces exact adherence to manga linework.
    2. **IP-Adapter (Image Prompt Adapter)**: Injects character appearance embeddings into cross-attention layers.
    3. **Regional Prompting / Seg-Guidance**: Ensures character colors don't bleed into background elements.
* **Color Harmonization**:
  * Post-process color pass with CLAHE (Contrast Limited Adaptive Histogram Equalization) and LAB color transfer for cross-panel chromatic uniformity.

### Phase 4: Timeline Scene Director & Keyframing (30-Frame Budget)
To convert a 3-page sequence into 30 frames, the engine supports two distinct timeline synthesis modes:

* **Mode A: Climax Action Continuity (Default)**
  * Selected focal panel from the 3 pages (e.g. the major slash, leap, or reveal).
  * 30 continuous frames generated via Image-to-Video diffusion centered on this action.
  * Budget: 16 native diffusion frames interpolated 2x to 32 frames (trimmed to 30).
* **Mode B: Multi-Panel Scene Montage (Narrative Arc)**
  * Shot 1 (Setup, Panel 1-2): 8 frames with subtle camera push-in.
  * Shot 2 (Action/Climax, Panel 3-4): 14 frames with dynamic motion diffusion.
  * Shot 3 (Reaction/Impact, Panel 5): 8 frames with slow-motion settle.
  * Total = 30 frames synced to a 24fps/30fps master clock.

### Phase 5: Image-to-Video (I2V) Motion Diffusion
* **Motion Backends Supported**:
  1. **AnimateDiff (SD1.5 / SDXL)** with Motion LoRA (Pan, Zoom, Tilt) + ControlNet Lineart & Depth.
  2. **Wan2.1-I2V / CogVideoX-5B / DynamiCrafter** for high-resolution fluid character transformations.
  3. **LivePortrait / Thin-Plate Spline (TPS)** fallback for talking heads or character facial animations under tight VRAM constraints.
* **Camera Trajectory**:
  * Automated camera path interpolation (dolly zoom, horizontal tracking, Dutch angle shake) computed from depth maps.

### Phase 6: Frame Interpolation & Temporal Post-Processing
* **Temporal Interpolation**:
  * **RIFE 4.6 (Real-Time Intermediate Flow Estimation)** or **FILM (Frame Interpolation for Large Motion)**.
  * Upscales low-fps latent output (e.g. 12-16 raw frames) to fluid 30 frames with edge-aware optical flow.
* **Temporal Deflickering & Anime Line Restoration**:
  * Temporal moving average filter in frequency space to eliminate diffusion shimmering.
  * **AnimeSR / Real-ESRGAN-anime** for crisp manga line preservation at 1080p/4K resolution.

---

## 4. Software Architecture & Directory Structure

```
manga2motion/
├── config/
│   ├── default_pipeline.yaml       # Master execution configuration
│   ├── color_palette.yaml          # Character & environment palette definition
│   └── motion_presets.yaml         # Camera curves and motion LoRA profiles
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── panel_extractor.py      # YOLOv8 panel segmentation & RTL ordering
│   │   ├── bubble_detector.py      # Text bubble & SFX masking
│   │   └── inpainter.py            # LaMa-based text removal & plate fill
│   ├── colorization/
│   │   ├── lineart_cleaner.py      # Screentone smoothing & line isolation
│   │   ├── palette_manager.py      # IP-Adapter & color memory store
│   │   └── colorizer_diffusion.py  # SDXL + ControlNet lineart color pipeline
│   ├── decomposition/
│   │   ├── sam2_segmenter.py       # Subject/foreground separation
│   │   └── depth_estimator.py      # Depth-Anything-V2 pipeline
│   ├── motion/
│   │   ├── timeline_director.py    # 30-frame budget & shot sequence planner
│   │   ├── i2v_diffusion.py        # AnimateDiff / Wan2.1 / SVD inference engine
│   │   └── camera_controller.py    # 2.5D parallax and camera trajectory curves
│   ├── postprocessing/
│   │   ├── interpolator.py         # RIFE / FILM 30-frame interpolation
│   │   ├── deflicker.py            # Temporal stabilization & color smoothing
│   │   └── video_encoder.py        # FFmpeg MP4 / ProRes / WebM export
│   └── ui/
│       ├── app.py                  # Gradio / FastAPI interactive web interface
│       └── components/             # Timeline scrubber, panel viewer, palette picker
├── tests/
│   ├── test_panel_ordering.py
│   ├── test_color_consistency.py
│   └── test_timeline_budget.py
├── scripts/
│   ├── download_models.sh          # Model weights downloader (Hugging Face)
│   └── run_cli.py                  # CLI entrypoint
├── requirements.txt
├── Dockerfile
└── plan.md
```

---

## 5. Data Flow & Timeline Specification

### 30-Frame Scene Timing Specification (`timeline.json`)

```json
{
  "scene_id": "manga_scene_001",
  "source_pages": ["page_01.png", "page_02.png", "page_03.png"],
  "total_target_frames": 30,
  "fps": 24,
  "color_profile": {
    "reference_image": "character_ref.png",
    "primary_character_palette": {
      "hair": "#C0392B",
      "skin": "#F5CBA7",
      "eyes": "#2980B9"
    }
  },
  "shots": [
    {
      "shot_id": 1,
      "source_panel": "page_01_panel_02",
      "frame_range": [0, 9],
      "frame_count": 10,
      "motion_type": "camera_dolly_in",
      "camera_speed": 1.2,
      "motion_prompt": "slow dramatic camera push-in, hair fluttering in wind, cinematic anime lighting"
    },
    {
      "shot_id": 2,
      "source_panel": "page_02_panel_01",
      "frame_range": [10, 21],
      "frame_count": 12,
      "motion_type": "dynamic_action_dash",
      "camera_speed": 2.5,
      "motion_prompt": "rapid forward motion, slashing sword attack, motion blur, glowing blade trail"
    },
    {
      "shot_id": 3,
      "source_panel": "page_03_panel_03",
      "frame_range": [22, 29],
      "frame_count": 8,
      "motion_type": "impact_freeze_settle",
      "camera_speed": 0.8,
      "motion_prompt": "shockwave expanding, dust particles settling, intense character gaze"
    }
  ]
}
```

---

## 6. Execution Modes & Hardware Strategy

To handle both high-end workstation setups and consumer/cloud environments:

| Tier | Target Hardware | Diffusion Engine | Optimization Techniques |
| :--- | :--- | :--- | :--- |
| **Tier 1: Cloud / Workstation** | NVIDIA RTX 3090 / 4090 / A100 (24GB+ VRAM) | SDXL + AnimateDiff + Wan2.1-I2V + RIFE | Full FP16, FlashAttention-2, xFormers, 1080p output |
| **Tier 2: Mid-Range GPU** | NVIDIA RTX 3060 / 4060 (8GB - 12GB VRAM) | SD1.5 AnimateDiff + Depth ControlNet | Sequential CPU Offloading, FP8 / INT8 quantization, 720p output |
| **Tier 3: Local CPU / Hybrid API** | Intel/AMD CPU + Modal / RunPod Serverless API | Local Panel Ingestion + Remote GPU Diffusion API | Panel extraction & inpainting locally, heavy diffusion offloaded via Webhook |

---

## 7. Implementation Roadmap & Milestones

### Milestone 1: Core Foundation & Dataset Pipeline (Week 1)
- [ ] Build project skeleton and package dependency management (`requirements.txt`, PyTorch, Diffusers).
- [ ] Implement `panel_extractor.py` using YOLOv8-manga with automatic RTL reading order sorting.
- [ ] Implement text bubble masking and inpainting using `LaMa` / OpenCV inpainting.
- [ ] Verify panel segmentation on a sample 3-page manga sequence.

### Milestone 2: Colorization & Style Consistency Engine (Week 2)
- [ ] Implement screentone suppression and bilateral line art filtering in `lineart_cleaner.py`.
- [ ] Set up SDXL + ControlNet Anime Lineart colorizer.
- [ ] Integrate IP-Adapter for single-reference character color adherence.
- [ ] Build histogram color matching to guarantee color consistency across panel boundaries.

### Milestone 3: Motion Generation & Keyframing (Week 3)
- [ ] Implement SAM 2 layer decomposition (`sam2_segmenter.py`) and depth map generation (`depth_estimator.py`).
- [ ] Implement `timeline_director.py` to calculate frame budgets (30 frames allocation across scene panels).
- [ ] Integrate I2V diffusion (AnimateDiff / SVD / Wan2.1) conditioned on colorized keyframes and motion prompts.
- [ ] Implement camera trajectories (pan, zoom, tilt) with depth-guided warping.

### Milestone 4: Frame Interpolation, Deflickering & Video Assembly (Week 4)
- [ ] Integrate RIFE v4.6 / FILM to interpolate raw diffusion frames up to the clean 30-frame target.
- [ ] Implement temporal deflicker algorithm (moving average in frequency/LAB space).
- [ ] Build `video_encoder.py` for automated MP4 / ProRes export with metadata tags.

### Milestone 5: Web UI & CLI Release (Week 5)
- [ ] Build Gradio WebUI with:
  - Drag-and-drop 3-page manga uploader.
  - Interactive panel sequence viewer and shot duration slider.
  - Color palette swatch picker & reference image uploader.
  - 30-frame interactive video scrubber and comparison view.
- [ ] Complete CLI interface (`python scripts/run_cli.py --pages page1.png page2.png page3.png --frames 30`).
- [ ] Write user guide, model setup scripts, and end-to-end unit tests.

---

## 8. Verification & Acceptance Criteria

1. **Panel Ordering Accuracy**: 100% correct panel sequence detection on standard 3-page right-to-left layout.
2. **Text Inpainting Quality**: No residual speech bubbles or dialogue clipping over character bodies.
3. **Color Uniformity**: Character hair, skin, and apparel color vectors maintain $<5\%$ delta-E color variance across all 30 frames.
4. **Frame Count & Timing**: Exactly 30 rendered frames exported at target framerate without frame drops or tearing.
5. **Temporal Smoothness**: Visual consistency with minimal flicker, evaluated via Warp Error / PSNR metrics across adjacent frames.
