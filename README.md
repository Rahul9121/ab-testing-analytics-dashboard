# A/B Testing Analytics Dashboard (Experimentation Platform)
Interactive experimentation dashboard built with **Streamlit + Plotly + SciPy + Statsmodels**. It supports uploaded experiment data, synthetic simulation, and Kaggle-compatible datasets to produce statistical decisions with interpretation.

## Creator
- **Rahul Ega** (`Rahul9121`)

## What this project does
- Upload A/B experiment CSV data and map columns interactively.
- Compute core experiment metrics:
  - p-value (two-proportion z-test)
  - confidence interval for absolute lift
  - absolute and relative conversion lift
  - recommended per-group sample size (80% power)
- Includes:
  - **Bundled Kaggle-style sample**
  - **Synthetic experiment generator**
  - Segment analysis and cumulative trend charts
  - Downloadable summary output
- Modern dark UI with KPI cards + interactive Plotly visuals.

## Recommended public dataset (Kaggle)
- Dataset URL: https://www.kaggle.com/datasets/rohankulakarni/ab-test-marketing-campaign-dataset
- You can download it via script:

```bash
python scripts/download_kaggle_dataset.py
```

Or use Kaggle CLI directly:

```bash
kaggle datasets download -d rohankulakarni/ab-test-marketing-campaign-dataset -p data/raw --unzip
```

## Project structure
- `app.py` - Streamlit dashboard entrypoint
- `analytics/data_utils.py` - parsing + synthetic generation
- `analytics/stats.py` - statistical test engine + interpretation layer
- `data/kaggle_style_ab_sample.csv` - bundled sample data
- `assets/style.css` - modern custom styling
- `.streamlit/config.toml` - Streamlit theme
- `Dockerfile` - containerized deployment

## Run locally
1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Launch app:

```bash
streamlit run app.py
```

## Run with Docker
Build image:

```bash
docker build -t ab-testing-dashboard .
```

Run container:

```bash
docker run --rm -p 8501:8501 ab-testing-dashboard
```

## Make this app live (online link) with Streamlit Community Cloud
1) Push this project to a GitHub repository.
2) Open https://share.streamlit.io/
3) Sign in with GitHub and click **New app**.
4) Select your repo + branch.
5) Set main file path to `app.py`.
6) Click **Deploy**.
7) Streamlit will generate a public URL like:
   - `https://<your-app-name>.streamlit.app`

## GitHub push flow (if creating repo manually)
```bash
git init
git add .
git commit -m "Initial commit: A/B testing analytics dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Notes
- Uploads can use flexible conversion values (`0/1`, `true/false`, `yes/no`).
- If your dataset has multiple variants, choose control and treatment in the UI.
- Store Kaggle credentials in `%USERPROFILE%\\.kaggle\\kaggle.json` before using Kaggle CLI downloads.
