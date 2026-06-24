<div align="center">

# 🎬 ReinaLook

**Turn your favorite film looks into a single Resolve LUT — from reference images, with real color science, 100% offline and free.**

[Install](#-install-windows-one-line) · [How it works](#-how-it-works) · [Usage](#-how-to-use-step-by-step) · [vs Higgsfield](#-reinalook-vs-higgsfield-ai-lut) · [Uninstall](#-uninstall)

</div>

---

## ✨ What is ReinaLook?

ReinaLook is a desktop app that generates a **33/65-point 3D `.cube` LUT** for **DaVinci Resolve**.
You feed it a few **reference images** (frames whose color grade you love), and it bakes that look —
warm/teal, filmic, moody, whatever — into one LUT you drop into your node graph.

It does the **technical conversion** (DaVinci Wide Gamut / Intermediate → Rec.709 Gamma 2.4) **and**
carries the **creative look** on a single **strength** dial. No subscription, no cloud, no uploading
your footage anywhere. It runs entirely on your machine.

> **The core promise:** the conversion is *sacred*. At strength 0 the LUT is your exact Resolve
> conversion, bit-for-bit. The look only ever blends **on top** — it can never break your base.

---

## 🧠 How it works

```
your footage ──▶ [Node 1: → DWG/DI] ──▶ [ReinaLook .cube] ──▶ Rec.709 + your look
                                          (replace Node 2, or sit between nodes)
```

1. **You give it reference images** — graded stills with the look you want.
2. ReinaLook studies their **color distribution** (tones, palette, saturation) using
   **optimal-transport color matching** — the same family of methods used in published color-transfer
   research, in the perceptual **Oklab** space.
3. It bakes that look over the **exact** DWG/DI → Rec.709 conversion and writes a `.cube`.
4. You load the `.cube` in Resolve. One node. Rides on every shot.

**Three ways to capture a look (pick what you have):**

| Mode | You provide | Best for |
|------|-------------|----------|
| **References** | graded stills of the look | the usual case |
| **Neutral + Graded (unpaired)** | a pool of your neutral frames + a pool of graded examples | better calibration to *your* footage |
| **Before/After pairs** | the same frames neutral **and** graded | cloning your *exact* grade, no guesswork |

**Look engines (all real, explainable math — no black box):**
- **Rich / MKL** — matches the references' mean + full color covariance (palette).
- **Rich / PDF** — Pitié *N-dimensional distribution transfer* (matches the whole color distribution).
- **Mid** — fast per-channel baseline.
- **Oklab** perceptual space, **Tone** (preserve brightness) and **Strength** dials on everything.

---

## ⚔️ ReinaLook vs Higgsfield AI LUT

Higgsfield's AI LUT generator is free and slick — but it's a **cloud, AI-image tool**. ReinaLook is a
**deterministic color-science tool that lives on your desktop**. Different philosophy:

| | **ReinaLook** | **Higgsfield AI LUT** |
|---|---|---|
| **Runs** | 100% **offline** on your PC | Cloud / browser, needs internet + account |
| **Your footage/photos** | **Never leave your machine** | Uploaded to their servers |
| **Method** | **Deterministic color science** (optimal transport, Oklab) — explainable, repeatable | AI model — a black box, results vary |
| **Same input → same output?** | **Always identical** | Not guaranteed |
| **Built for Resolve** | **Yes** — exact DWG/DI → Rec.709 base, replace Node 2 or sit between nodes | Generic LUT |
| **Conversion accuracy** | **Bit-exact** base (your real Resolve CST); strength 0 = untouched | N/A |
| **Clone *your* exact grade** | **Yes** (before/after pairs mode) | No |
| **Resolution** | 33 **or 65-point** cubes | Typically fixed |
| **Cost / limits** | **Free, unlimited, no login** | Free tier, account, rate limits |
| **Privacy** | Total | Cloud ToS apply |

**Use Higgsfield** if you want a quick AI vibe with no setup and don't mind the cloud.
**Use ReinaLook** if you want **accuracy, privacy, repeatability, and a LUT engineered for your exact
Resolve pipeline** — that you can tune, clone, and trust shot to shot.

---

## ⬇️ Install (Windows, one line)

Open **PowerShell** and paste this one line:

```powershell
irm https://raw.githubusercontent.com/QuagKhai003/ReinaLook/main/install.ps1 | iex
```

That single command will:
1. Download the latest **ReinaLook.exe** from GitHub Releases into `%LOCALAPPDATA%\ReinaLook`.
2. Create a **Start Menu** and **Desktop** shortcut.
3. **Launch the app.**

No admin rights, no Python, no extra installs needed. (≈155 MB download; everything is bundled.)

> **Windows SmartScreen** may warn on first run (unsigned app). Click **More info → Run anyway**.
> After installing, just open **ReinaLook** from the Start Menu or Desktop any time.

<details>
<summary>Run from source instead (developers)</summary>

```bash
git clone https://github.com/QuagKhai003/ReinaLook.git
cd ReinaLook
pip install -e ".[gui]"
reinalook-gui          # or:  python -m lutgen.app
```
</details>

---

## 🎚️ How to use (step by step)

### In ReinaLook
1. **Open ReinaLook** (Start Menu / Desktop shortcut).
2. **Pick a Mode** (top-left):
   - **References (graded only)** — the simplest. You'll add stills of the look you want.
   - **Neutral + Graded (unpaired)** — add a pool of *your* neutral frames **and** a pool of graded
     examples (any counts, different scenes OK).
   - *(Before/After pairs is available via the command line for exact-grade cloning — see below.)*
3. **Add reference images:** click **+ Add references…** and select your look stills (JPG/PNG/TIFF).
4. **Choose the look engine** (sensible defaults are fine):
   - **Placement:** `Replace CSTout` (most accurate — swaps your Node 2) or `Between CSTs` (sits
     between Node 1 and Node 2, keeps both).
   - **Fitter:** `Rich` (recommended) or `Mid` (simple).
   - **Method:** `mkl` (palette) or `pdf` (richest, full distribution).
   - **Space:** `oklab` (perceptual, recommended) or `rgb`.
5. **Set the dials:**
   - **Tone** — lower keeps your footage's brightness; higher also matches the references' exposure.
   - **Strength** — how strong the look is (0 = your original, 1 = full look).
6. **(Optional) Load a preview still:** click **Load preview still (DWG/DI frame)…** and pick a
   **flat/log frame from your project** (a clip with Node 1 on, Node 2 off) to see a real *before /
   after* on the right. It appears instantly.
7. **Click `Compute preview`** to render the look (a progress % shows; controls grey out while it
   works). Tweak dials and re-Compute until you like it.
8. **Click `Export .cube…`** and save the LUT. (Optionally **Save preset…** to reuse the recipe.)

### In DaVinci Resolve
1. Put the `.cube` somewhere Resolve sees it, or **Color page → LUTs → Open LUT Folder**, drop it in,
   right-click the LUT panel → **Refresh**.
2. **Replace CSTout** placement → use the LUT **in place of your Node 2** (the DWG/DI → Rec.709 CST).
3. **Between CSTs** placement → add a node **between** Node 1 and Node 2 and apply the LUT there;
   keep both CST nodes.
4. Keep Node 1 and any of your own adjustments. Tweak strength by re-exporting if needed. Done — the
   look rides on every shot in the timeline.

### Command line (optional, for power users)
```bash
# References → look cube (Rich, Oklab, full distribution):
reinalook render --refs r1.png r2.png r3.png --fitter rich --method pdf --space oklab ^
                 --strength 0.8 --tone 0.5 --placement node2 --out look.cube

# Unpaired pools (your neutral footage + graded examples):
reinalook render --source neutral1.png neutral2.png --refs graded1.png graded2.png --out look.cube

# Clone your EXACT grade from before/after frames (same frame, grade off vs on):
reinalook render-pairs --before n1.png n2.png --after g1.png g2.png --out my_grade.cube
```

---

## 🗑️ Uninstall

One line in PowerShell removes **everything** ReinaLook installed (app, shortcuts, cache):

```powershell
irm https://raw.githubusercontent.com/QuagKhai003/ReinaLook/main/uninstall.ps1 | iex
```

This deletes:
- the app folder `%LOCALAPPDATA%\ReinaLook` (the exe + install record),
- the **Start Menu** and **Desktop** shortcuts,
- ReinaLook's PyInstaller temp-extract cache in `%TEMP%`.

Your own exported `.cube` / preset `.json` files (wherever *you* saved them) are **left untouched**.

<details>
<summary>Prefer to do it by hand?</summary>

1. Delete the folder `%LOCALAPPDATA%\ReinaLook` (paste that path into File Explorer).
2. Delete the **ReinaLook** shortcut from your Desktop and Start Menu.
3. (Optional) Delete any `_MEI*` folders in `%TEMP%` left by the app.
</details>

---

## ❓ FAQ

- **Do I need DaVinci Resolve Studio?** No — the free version loads `.cube` LUTs.
- **Does it need my source footage?** No. It synthesizes the DWG/DI input domain; you only need
  reference images (and optionally one flat preview frame).
- **Why does the base look exact but Replace softens a hair at 300% zoom?** A LUT samples a continuous
  conversion at grid points; 65-point minimizes it. Use 65-point base / `Between CSTs` if you need
  maximum sharpness.
- **Is my data private?** Completely. Nothing is uploaded; there's no network call at all.

---

## 🛠️ For maintainers — cut a release

The installer pulls a prebuilt `ReinaLook.exe` from GitHub Releases:

```bash
pip install -e ".[gui,package]"
pyinstaller --noconfirm packaging/ReinaLook.spec      # → dist/ReinaLook.exe
```
Then create a GitHub Release (e.g. tag `v1.0.0`) and upload `dist/ReinaLook.exe` as an asset named
exactly **`ReinaLook.exe`**. The one-line installer always fetches the latest release.

---

<div align="center">
Made for colorists who want the look <b>and</b> the control. • Free • Offline • Deterministic
</div>
