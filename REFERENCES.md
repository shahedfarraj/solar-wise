# References

All sources used in this project — data, domain assumptions, and the tools/libraries it's built on. Useful for the report's references section and for anyone continuing the project.

---

## Data Sources

- **NASA POWER** — the weather/solar dataset this whole project is built on.
  [power.larc.nasa.gov](https://power.larc.nasa.gov/)
- **NASA POWER API docs** — the exact endpoint used to pull daily data.
  [power.larc.nasa.gov/docs/services/api/temporal/daily](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
- **OpenStreetMap Nominatim** — used to turn addresses into coordinates, and to check whether a point is really inside Jordan's border.
  [nominatim.openstreetmap.org](https://nominatim.openstreetmap.org/)
- **Nominatim usage policy** — the rate-limit rules (1 request/second) this project follows when calling it.
  [operations.osmfoundation.org/policies/nominatim](https://operations.osmfoundation.org/policies/nominatim/)

---

## Domain / Industry Standards (assumptions used in the model)

- **Standard solar panel sizes and wattages** — panel dimensions and wattage assumption (450W monocrystalline).
  [The Green Watt](https://www.thegreenwatt.com/standard-solar-panel-sizes-and-wattages-dimensions/)
- **Solar panel size and wattage guide** — panel efficiency assumption (21%).
  [Palmetto](https://palmetto.com/solar/choosing-the-right-solar-panel-size-and-wattage)
- **Middle East Solar Compliance 2026: Jordan** — residential system size range (1–5.4 kWp), Net Billing regulation (Bylaw 58, 2024).
  [SurgePV](https://www.surgepv.com/solar-compliance/middle-east)
- **JEPCO official tariff page** — the tiered residential electricity pricing used for the JOD savings calculation.
  [JEPCO](https://www.jepco.com.jo/ar/Home/فئات-وشرائح-تعرفة-الكهرباء)

---

## Tools & Libraries

- **Python** — [python.org](https://www.python.org/)
- **Jupyter Notebook** — used to build and run `solar_worth_it.ipynb`.
  [jupyter.org](https://jupyter.org/)
- **pandas / NumPy** — data handling and numeric computation.
  [pandas.pydata.org](https://pandas.pydata.org/) · [numpy.org](https://numpy.org/)
- **scikit-learn** — Linear Regression, Random Forest, train/test tooling, metrics.
  [scikit-learn.org](https://scikit-learn.org/)
- **XGBoost** — the best-performing model in this project.
  [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
- **Plotly** — every chart in the notebook and the app.
  [plotly.com/python](https://plotly.com/python/)
- **Streamlit** — the web app framework the whole app is built with.
  [docs.streamlit.io](https://docs.streamlit.io/)
- **Folium** — the interactive map used for "pick your location on a map."
  [python-visualization.github.io/folium](https://python-visualization.github.io/folium/)
- **streamlit-folium** — lets a Folium map be clicked inside a Streamlit app.
  [github.com/randyzwitch/streamlit-folium](https://github.com/randyzwitch/streamlit-folium)
- **streamlit-js-eval** — powers the "Detect my location" browser GPS feature.
  [github.com/aghasemi/streamlit_js_eval](https://github.com/aghasemi/streamlit_js_eval)
- **joblib** — saving/loading the trained model and features for the app.
  [joblib.readthedocs.io](https://joblib.readthedocs.io/)

---

## Deployment

- **Git** — version control, used to push the project to GitHub.
  [git-scm.com](https://git-scm.com/)
- **GitHub** — where the code is hosted.
  [github.com](https://github.com/)
- **Streamlit Community Cloud** — free hosting for the live app.
  [share.streamlit.io](https://share.streamlit.io/)
