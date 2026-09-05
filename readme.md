# 📈 Métricas de Valuación de Activos Financieros

Dashboard en **Streamlit** para evaluar el desempeño y el riesgo de activos de
renta variable a partir de precios de **Yahoo Finance**, con estética bursátil
FINTECH (azul / negro, tipografía gris Arial).

> Mercados Financieros · Renta Variable · Uso académico

---

## 1. Qué calcula

| # | Indicador | Fórmula |
|---|-----------|---------|
| 1 | Rentabilidad anualizada | $(V_f / V_i)^{1/n} - 1$ |
| 2 | Volatilidad anualizada | $\sigma \sqrt{n}$ |
| 3 | Índice Sharpe | $(R_p - R_f) / \sigma_p$ |
| 4 | Coeficiente de correlación de Pearson | $\rho(R_i, R_m)$ |
| 5 | BETA | $\mathrm{Cov}(R_i, R_m) / \sigma_m^2$ |
| 6 | Índice Traynor | $(R_a - R_f) / \beta_a$ |
| 7 | CAPM | $R_f + \beta_i (R_m - R_f)$ |
| 8 | Alpha | $R_i - [R_f + \beta_i (R_m - R_f)]$ |
| 9 | Valor $z$ | $z_\alpha$ de la distribución normal estándar |
| 10 | VaR % y VaR $ | $\mathrm{VaR}_\alpha = \mu + z_\alpha \sigma$ |

Más: matriz de correlación, regresión OLS de cada activo contra el índice de
referencia, evolución de precios base 100, distribución de retornos con el corte
del VaR y exportación de resultados a CSV.

---

## 2. Fuentes de datos

| Dato | Origen |
|------|--------|
| Precios de cierre / cierre ajustado | Yahoo Finance (`yfinance`) |
| Índice bursátil de referencia | Yahoo Finance (`^GSPC`, `^MXX`, `^IBEX`, …) |
| Tasa libre de riesgo | Instrumento soberano de corto plazo del país de origen del activo. Para EE.UU. se descarga automáticamente `^IRX` (T-Bill 13 semanas) o `^TNX` (Nota 10 años); para el resto de países se precarga un valor de referencia **editable** en el panel lateral. |

> ⚠️ Verifica siempre la tasa libre de riesgo antes de interpretar Sharpe,
> Traynor, CAPM y Alpha: los valores precargados son referencias, no cotizaciones
> en tiempo real.

---

## 3. Inputs (panel lateral)

- **Número de activos a valuar** (1 a 8) y sus **tickers**
- **Índice bursátil de referencia** (catálogo o ticker libre)
- **País de origen** → tasa libre de riesgo anual (%)
- **Plazo a calcular**: 5 días · 3 meses · 6 meses · 12 meses · 1 año · 5 años
- **Periodicidad de precios**: diaria · semanal · mensual
- **Serie de precios**: cierre ajustado o cierre simple
- **Monto de capital a invertir** para el VaR
- **Intervalo de confianza**: 90 % · 95 % · 97.5 % · 99 %
- **Plazo para VaR**: 1 · 5 · 10 · 21 · 63 · 126 · 252 días
- **Ponderación del portafolio**: pesos iguales o definidos por el usuario

---

## 4. Estructura del repositorio

```
valuacion-activos/
├── app.py                  # Aplicación Streamlit
├── requirements.txt        # Dependencias
├── readme.md               # Este archivo
└── .streamlit/
    └── config.toml         # Tema azul / negro
```

---

## 5. Ejecución local

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

La app abre en `http://localhost:8501`.

---

## 6. Publicación en GitHub

```bash
git init
git add app.py requirements.txt readme.md .streamlit/config.toml
git commit -m "Dashboard de métricas de valuación de activos"
git branch -M main
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```

## 7. Despliegue en Streamlit Community Cloud

1. Entra a **https://share.streamlit.io** e inicia sesión con GitHub.
2. **New app** → selecciona el repositorio, la rama `main` y el archivo `app.py`.
3. **Deploy**. La primera compilación instala `requirements.txt` (1–3 min).
4. La app queda publicada en `https://<nombre>.streamlit.app`.

No se requieren *secrets* ni claves de API: Yahoo Finance se consulta de forma
anónima.

---

## 8. Notas metodológicas y supuestos

- Retornos **aritméticos** sobre la periodicidad seleccionada.
- Anualización: 252 periodos (diaria), 52 (semanal), 12 (mensual).
- VaR **paramétrico** bajo normalidad; escalado al horizonte con $\mu t$ y
  $\sigma\sqrt{t}$. No captura colas gruesas ni eventos extremos.
- $R_f$ constante durante la ventana; sin costos de transacción ni impuestos.
- En ventanas cortas (5 días, 3 meses) la anualización amplifica el ruido
  estadístico: los resultados son indicativos.
- Los precios se descargan en la divisa de cotización de cada ticker; mezclar
  mercados introduce riesgo cambiario no modelado.
- Caché de precios de 15 minutos (`st.cache_data`).

---

## 9. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| «No se obtuvieron precios» | Ticker inexistente o mercado cerrado sin histórico | Verifica el símbolo exacto en Yahoo Finance (ej. `WALMEX.MX`) |
| Pocas observaciones | Periodicidad mensual con plazo corto | Usa periodicidad diaria o amplía el plazo |
| Beta o Traynor vacíos | Varianza del índice nula o serie muy corta | Amplía la ventana de análisis |
| Rentabilidad anual extrema | Anualización de una ventana de días | Usa plazos de 12 meses o más |

---

## 10. Aviso

Herramienta con **fines académicos**. No constituye una recomendación de
inversión ni asesoría financiera. Los datos provienen de Yahoo Finance y pueden
presentar retrasos o errores.
