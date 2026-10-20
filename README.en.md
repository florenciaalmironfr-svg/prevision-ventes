# 🛍️ Black Friday Sales Simulator · 2025 Sales Forecast

[🇫🇷 Français](README.md) · 🇬🇧 English

🔗 **[Live demo](https://prevision-ventes-almiron.sreamlit.app)** · Try the simulator directly in your browser.

Application that **forecasts daily November 2025 sales** for 24 products of a sports retailer and lets users **simulate pricing and competition scenarios** to support Black Friday commercial decisions.

> Project built as part of the **DS4B** Data Science + AI training, extended with my own analyses and improvements (see [My contributions](#-my-contributions)).

<!-- Add a screenshot of the simulator here: docs/img/simulateur.png -->
![Simulator](docs/img/simulateur.png)

---

## 🎯 Business problem

A sports retailer needs to anticipate sales during its key season (late October to late November) to decide on **discounts**, **stock** and **price positioning** against Amazon, Decathlon and Deporvillage.

Questions the application answers:
- How many units will each product sell, day by day, in November 2025?
- What is the impact of a discount applied only on Black Friday?
- What happens if competitors lower or raise their prices by 5%?
- Does higher revenue translate into higher margin?

## 📦 Data

| Dataset | Content |
|---|---|
| Sales 2021-2024 | 24 products, from October 25 to November 30 each year (3,552 records): base price, selling price, units, revenue, category |
| Competition | Daily prices from Amazon, Decathlon and Deporvillage per product |
| Inference 2025 | Late October with actual sales (lag initialization) and November to forecast |

## 🛠️ Approach

| Step | Content |
|---|---|
| **Data quality** | Types, missing values, duplicates, consistency check `revenue = price × units` |
| **Exploratory analysis** | Seasonality, weekdays, categories, top products, price distribution vs Amazon |
| **Feature engineering** | Spanish calendar (public holidays, weekends, start/end of month), Black Friday and Cyber Monday, lags 1 to 7 and 7-day moving average per product, discount rate, average competitor price and price ratio, One Hot Encoding |
| **Validation** | Train on 2021-2023, test on 2024 (time-based split) |
| **Model** | `HistGradientBoostingRegressor` with conservative parameters, compared with a baseline (mean) |
| **Diagnostics** | Error by product and by period, actual vs predicted, permutation feature importance |
| **2025 forecast** | Final model trained on 2021-2024, **recursive** day-by-day forecasting |
| **Application** | Streamlit, French interface |

**Stack:** Python · pandas · scikit-learn · seaborn · Streamlit · GitHub Codespaces · GitHub Copilot (agent mode)

## 📊 Results (validation on 2024)

| | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 3.67 | 6.69 | 0.00 |
| **Final model** | **1.05** | **2.23** | **0.89** |

The model reduces error by **71%** compared with the simple method.

**Simulation example** (Adidas Ultraboost 23, November 2025): a −15% discount applied only on Black Friday yields **+10.6% units** and **+8.7% revenue**, with a peak of about 62 units on November 28, consistent with historical data.

## 💡 My contributions

| Contribution | Impact |
|---|---|
| **Lags computed per product** (instead of across all products) | Removed a data leakage issue that artificially inflated metrics |
| **Fixed the 2024 Black Friday date** (the "fourth Friday" rule flagged November 22 instead of November 29) | MAE improved from 1.44 to 1.05 (**−27%**) by fixing a single date |
| **Detected confounding between discounts and Black Friday** | Explains why the model overestimates the effect of discounts outside the event |
| **Experiment to separate both effects** | Documented negative result: a data limitation, not a method limitation |
| **Enhanced application** | Discount by day, French interface, price / volume variance analysis, margin impact, model-limitation alerts |

## 🔎 Key findings

| # | Finding | Consequence |
|---|---|---|
| 1 | The model overestimates the effect of small discounts (discount days without an event) | Discount simulations must be read with caution |
| 2 | The "fourth Friday" rule placed Black Friday 2024 on November 22 | Fixed: MAE −27% |
| 3 | Lags computed across all products create data leakage | Lags computed per product and per year |
| 4 | Error depends more on the product than on the time of the month | Nike Pegasus is the least accurately forecast product, Trek Marlin the most |
| 5 | Discount is the most important feature, ahead of Black Friday | Confirms the discount / event confounding |
| 6 | A discount applied all month is amplified by recursive forecasting ("snowball effect") | Recommended use: discount limited to Black Friday |
| 7 | The model does not extrapolate beyond the historical price range (−15% to +3% of base price) | A +40% increase is treated as +3%; the app shows an alert |

## 🖥️ The application

- Product selection, price adjustment (−50% to +50%) and choice of days: Black Friday only, Black Friday week, whole month or custom selection
- Three competition scenarios: current prices, −5% and +5%
- Key indicators, daily chart with Black Friday highlighted, detailed table
- Economic impact: revenue with and without adjustment and **price / volume variance analysis** (volume effect, price effect and consistency check), following management control practice
- **Waterfall chart**: revenue without adjustment → volume effect → price effect → revenue with adjustment
- Margin impact if a unit cost is provided
- **Alerts**: atypical result (revenue variance above +100%) and scenario outside the historical price range

## 📐 Assumptions

1. 2025 behavior will be similar to 2021-2024.
2. 2025 competitor prices are known; scenarios shift all of them by the same percentage.
3. Each product's base price is constant over the period.
4. Sales = demand: no stockouts.
5. Products are independent: no cannibalization.
6. Only Spanish national public holidays are considered.
7. Black Friday = day after the fourth Thursday of November.
8. Without adjustment, the simulated selling price equals the base price (historical daily variations are not reproduced).
9. Forecast sales are rounded to the nearest unit; revenue is computed on rounded units.
10. Revenue is computed on displayed prices, likely VAT-inclusive; a financial analysis should convert it to net of VAT.

## ⚠️ Limitations

| Limitation | Consequence |
|---|---|
| Data limited to the Black Friday season (≈ 37 days per year) | The model does not apply to other months |
| Discounts historically concentrated on Black Friday | Overestimates the effect of off-event discounts, underestimates the peak without discount |
| Recursive forecasting | Errors accumulate, especially with long discounts |
| Tree-based model | No extrapolation beyond the historical price range (−15% to +3%) |
| No confidence intervals | The app gives a single value, not a range |
| Small volume: 24 products, 4 seasons | Product-level results are fragile |
| Training-course data | The project's value lies in the method, not the figures |

**Recommended use:** simulate discounts during the event period, as in real commercial practice, and read results as indicative scenarios.

## 🗺️ Next steps

1. Confidence intervals (optimistic and pessimistic scenarios)
2. Price tests outside Black Friday to measure the true effect of discounts
3. Real product costs to optimize margin
4. Cannibalization features between products of the same category
5. Deployment on Streamlit Community Cloud

## 📁 Project structure

```
├── app.py                         # Streamlit application
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── entrenamiento/         # Sales and competition 2021-2024
│   │   └── inferencia/            # 2025 data
│   └── processed/                 # Transformed data
├── models/
│   └── modelo_final.joblib        # Trained model
├── notebooks/
│   ├── Entrenamiento.ipynb        # Quality, EDA, features, model, findings
│   └── forecasting.ipynb          # 2025 data preparation
└── .github/
    └── copilot-instructions.md    # Rules for the AI assistant
```

## ▶️ Run the application

**With GitHub Codespaces:** **Code → Codespaces → Create codespace**, then in the terminal:

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Locally:**

```bash
git clone https://github.com/florenciaalmironfr-svg/prevision-ventes.git
cd prevision-ventes
pip install -r requirements.txt
streamlit run app.py
```

## 👩‍💼 Author

**Florencia Almiron** · Management Controller | Finance IS Business Analyst | Data & BI | Applied AI

*Connecting business, processes and data for better decisions.*
