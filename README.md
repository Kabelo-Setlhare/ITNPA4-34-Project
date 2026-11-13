# ITNPA Project 1: Researchly Research Abstract Classification and Document Mining

| Folder | Contents |
|---|---|
| `docs/` | Project documentation (Questions 1 to 4) and the interactive pyLDAvis map (`lda_intertopic_map.html`) |
| `data/` | `research_abstracts.json` (labelled abstracts), `research_corpus/` (5 articles), `abstracts_clean.csv` (Q1 output) |
| `notebooks/` | Q1 preprocessing and baseline, Q2 models/tuning/deployment, Q3 document mining and summarisation, Q4 LDA and pyLDAvis |
| `deployment/` | Streamlit app and a README with easy deployment steps (Questions 1 and 2) |

## Run order
1. `pip install -r requirements.txt`
2. Run the notebooks in order (Q1, Q2, Q3, Q4). Q2 reads the file Q1 saves.
3. Follow `deployment/README.md` to launch the app.
