# Data notes

- Source used by the supplied exercise: https://raw.githubusercontent.com/avikumart/Road-Traffic-Severity-Classification-Project/main/RTA%20Dataset.csv
- Expected SHA-256: `975716a9e7014e5f49eaca3cf7ba282f3fa7a36db3a30fca79d7cb42e06df053`.
- Reviewed snapshot: 12,316 rows, 32 columns and three labels. The source material describes Addis Ababa police records, 2017–2020; this attribution is not an independent audit of the original collection process.
- Original collection consent, licensing chain and mirror redistribution permissions have not been independently established. Raw CSV is ignored by Git and omitted from the downloadable project bundle. Downloading for this study does not establish production-use rights.
- The repository includes derived aggregate metrics, schema category names and a few scene-only training examples. It excludes names, contact information, class rosters, proprietary course files and raw casualty rows.
- `artifacts/split_indices.json` stores row indices for reproducing this exact snapshot; it is generated locally and not pushed. Raw data and model files are ignored to keep the repository lightweight and avoid silently redistributing the source dataset.
