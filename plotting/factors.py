import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import ipywidgets as widgets
from ipywidgets import Layout, HBox, VBox
from IPython.display import display, clear_output

class FactorRiskAnalyzer:
    """
    A consolidated class that wraps key analytics for a FactorRiskModel:
      - Forecasting portfolio volatility
      - Scenario factor shocks (with an interactive slider UI)
      - Plotting factor risk decomposition (Plotly)
      - Plotting factor returns (using top factor exposures from the risk decomposition)
      - Forward-looking simulation ("cone") of future portfolio PnL (with nearest-PD fix)
      - Comparison of risk forecasts with another FactorRiskModel
      - Treemap of factor exposures, including an approximate multi-group approach
    """

    def __init__(self, risk_model, portfolio_weights=None):
        """
        Parameters
        ----------
        risk_model : FactorRiskModel
            Must support:
              - get_cov(), get_beta(), get_idio(), get_factor_returns()
              - decompose_risk(portfolio) -> DataFrame with columns like
                [exposure, vol_pct, vol_contri, ...].
        portfolio_weights : pd.Series, optional
            index=tickers, values=positions or weights.
        """        
        self.risk_model = risk_model
        self.portfolio = portfolio_weights
        self.name = risk_model.name or ''

        # Store the last risk decomposition (for top factor info, etc.)
        self._last_risk_report = pd.DataFrame()
        self._last_top_factors = []

        # For scenario slider UI
        self._scenario_sliders = {}
        self._scenario_output = None
        self._scenario_hbox = None
        self._sliders_vbox = None

    # ---------------------------------------------------------------------
    # 1) CORE METHODS
    # ---------------------------------------------------------------------
    @staticmethod
    def cut_down_model(model,portW = None):
        if isinstance(portW,pd.Series):
            if len(model.beta.index) != len(portW.index):
                model.beta = model.beta.loc[portW.index]
            if len(model.idio_vol.index) != len(portW.index):
                model.idio_vol = model.idio_vol.loc[portW.index]
        return model
        
    def forecast_portfolio_vol(self, portfolio_weights=None) -> float:
        """
        Forecast the portfolio's annualized volatility using factor + idiosyncratic risk.
        Returns a float (e.g. 0.12 => 12%).
        """
        pw = self._check_portfolio(portfolio_weights)
        riskModelCut = self.cut_down_model(self.risk_model,pw)

        cov = riskModelCut.get_cov()
        beta = riskModelCut.get_beta()
        idio = riskModelCut.get_idio()

        factor_expo = pw.dot(beta)
        factor_var = factor_expo.dot(cov).dot(factor_expo)
        idio_var = (pw**2 * idio**2).sum()
        return float(np.sqrt(factor_var + idio_var))

    def scenario_impact(self, factor_shocks: dict, portfolio_weights=None) -> float:
        """
        Approximate portfolio return given factor shocks in multiples of stdev.
        e.g. factor_shocks={'Growth':-2.0, 'Value':+1.0}
        => "Growth factor down 2 stdev, Value factor up 1 stdev."
        """
        pw = self._check_portfolio(portfolio_weights)
        riskModelCut = self.cut_down_model(self.risk_model,pw)
        factor_cov = riskModelCut.get_cov()
        shock_vector = pd.Series(0.0, index=factor_cov.index)

        for fct, shock_sd in factor_shocks.items():
            if fct in shock_vector.index:
                factor_std = np.sqrt(factor_cov.loc[fct, fct])
                shock_vector[fct] = shock_sd * factor_std

        beta = riskModelCut.get_beta()
        factor_expo = pw.dot(beta)
        return float(factor_expo.dot(shock_vector))

    # ---------------------------------------------------------------------
    # 2) FACTOR RISK CONTRIBUTION & FACTOR RETURNS
    # ---------------------------------------------------------------------

    def plot_factor_risk_contribution(self, 
                                      portfolio_weights=None, 
                                      pct_or_abs='pct', 
                                      top_x=None):
        """
        Decompose the portfolio's risk and plot factor contributions in an interactive bar chart.
        Also stores the results in _last_risk_report and _last_top_factors.
        """
        pw = self._check_portfolio(portfolio_weights)
        riskModelCut = self.cut_down_model(self.risk_model,pw)
        risk_report = riskModelCut.decompose_risk(pw)
        if risk_report.empty:
            print("No risk decomposition available (maybe zero portfolio?).")
            return

        # Always store a "master" sorted version for get_risk_report
        master_sorted = risk_report.sort_values('vol_contri', ascending=False, key=abs)
        self._last_risk_report = master_sorted.copy()

        field = 'vol_pct' if pct_or_abs == 'pct' else 'vol_contri'
        rr_plot = risk_report.sort_values(by=field, ascending=False, key=abs)

        # Save top factor list excluding "Specific"
        self._last_top_factors = [f for f in rr_plot.index if f != "Specific"]

        # If requested, keep only top X
        if top_x is not None and top_x > 0 and top_x < len(rr_plot):
            rr_plot = rr_plot.head(top_x)

        rr_plot["color"] = ["red" if idx == "Specific" else "blue" for idx in rr_plot.index]
        df_plot = rr_plot.reset_index().rename(columns={"index": "Factor"})

        title_str = f"Factor Risk Contribution ({'%' if pct_or_abs=='pct' else 'abs'})"
        fig = px.bar(
            df_plot,
            x="Factor", y=field,
            title=title_str + f" - {self.name}",
            color="color",
            labels={"Factor": "Factor", field: "Risk Contribution"},
            color_discrete_map={"red": "red", "blue": "blue"},
            template="plotly_white"
        )

        def rename_legend(trace):
            trace.update(name="Specific" if trace.name=="red" else "Factor")
        fig.for_each_trace(rename_legend)

        if field == 'vol_pct':
            yaxis_fmt = ".1%"
        else:
            yaxis_fmt = None

        fig.update_traces(marker_line_width=0, marker_line_color="black")
        fig.update_layout(
            xaxis_title="Factor",
            yaxis_tickformat=yaxis_fmt,
            hovermode="x unified",
            legend_title="Factor Type",
            xaxis=dict(categoryorder='total descending')
        )
        fig.update_xaxes(categoryorder='total descending')
        fig.show()

    def plot_factor_returns(self, top_x=None, portfolio_weights=None):
        """
        Plot the cumulative factor returns for the top factors that matter
        to the current portfolio, or fallback if no decomposition is stored.
        """
        if self._last_risk_report.empty:
            print("No stored risk_report from plot_factor_risk_contribution. Attempting fallback.")
            riskModelCut = self.cut_down_model(self.risk_model,self.portfolio)
            factor_rets = riskModelCut.get_factor_returns()
            if factor_rets.empty:
                print("No factor returns data stored in the model.")
                return
            fallback_top_x = top_x if top_x else 10
            var_sorted = factor_rets.var().sort_values(ascending=False)
            fallback_factors = var_sorted.index[:fallback_top_x]
            self._plot_cumulative_factor_returns(fallback_factors, factor_rets)
            return

        factor_rets = self.risk_model.get_factor_returns()
        if factor_rets.empty:
            print("No factor returns data stored in the model.")
            return

        factors_all = self._last_top_factors
        if not factors_all:
            print("No overlap between top factors and factor_returns columns.")
            return

        if top_x is not None and top_x < len(factors_all):
            factors_all = factors_all[:top_x]

        meaningful_factors = [f for f in factors_all if f in factor_rets.columns]
        self._plot_cumulative_factor_returns(meaningful_factors, factor_rets)

    def get_risk_report(self, top_x=None) -> pd.DataFrame:
        """
        Return the last stored risk report (sorted by abs vol_contri).
        If top_x is specified, returns only that many rows.
        """
        if self._last_risk_report.empty:
            pw = self._check_portfolio(None)
            rr = self.risk_model.decompose_risk(pw)
            rr_sorted = rr.sort_values('vol_contri', ascending=False, key=abs)
            self._last_risk_report = rr_sorted

        df_out = self._last_risk_report.copy()
        if top_x is not None and top_x < len(df_out):
            df_out = df_out.head(top_x)
        return df_out

    # ---------------------------------------------------------------------
    # 3) FORWARD-LOOKING CONE
    # ---------------------------------------------------------------------

    def plot_forward_cone(self, 
                          hist_pnl: pd.Series, 
                          horizon_days=20, 
                          n_scenarios=1000, 
                          portfolio_weights=None,
                          quantiles=(10, 25, 50, 75, 90)):
        """
        Plot a forward-looking cone around the portfolio PnL using a random simulation approach,
        with a nearest-PD fix if the factor covariance is not strictly posdef.
        """
        pw = self._check_portfolio(portfolio_weights)
        if hist_pnl.empty:
            print("Empty hist_pnl. Cannot create cone plot.")
            return

        df_quantiles = self._simulate_forward_cone(
            pw, hist_pnl, horizon_days, n_scenarios, quantiles
        )
        self._plot_cone(hist_pnl, df_quantiles)

    # ---------------------------------------------------------------------
    # 4) COMPARE WITH ANOTHER MODEL
    # ---------------------------------------------------------------------

    def compare_to_other_model(self, other_risk_model, portfolio_weights=None):
        """
        Compare this model vs. another FactorRiskModel for the same portfolio.
        Show side-by-side total volatility forecast and factor decomposition difference.
        """
        pw = self._check_portfolio(portfolio_weights)
        my_vol = self.forecast_portfolio_vol(pw)
        temp_analyzer = FactorRiskAnalyzer(other_risk_model, pw)
        other_vol = temp_analyzer.forecast_portfolio_vol()
        otherModelName = temp_analyzer.name or 'OtherModel'
        thisModelName = self.risk_model.name or 'ThisModel'
        summary = pd.DataFrame({
            "Model": [thisModelName, otherModelName],
            "ForecastVol": [my_vol, other_vol]
        })

        print("Comparison of total forecast volatility:")
        #print(summary)
        return summary

    # ---------------------------------------------------------------------
    # 5) INTERACTIVE SCENARIO SLIDER (no vertical scrollbar)
    # ---------------------------------------------------------------------

    def interactive_scenario_slider(self, 
                                    factor_list=None, 
                                    top_x=None, 
                                    portfolio_weights=None):
        """
        Creates an interactive slider UI in a single horizontal layout:
          - A VBox of sliders on the LEFT
          - A single bar chart on the RIGHT
        As you move sliders, we update the chart in-place (only one figure).
        
        We do NOT set a max-height scroll, so if you have many factors
        it will just extend vertically. 
        """
        pw = self._check_portfolio(portfolio_weights)

        # 1) Determine which factors to show
        if factor_list is None:
            if self._last_risk_report.empty:
                rr = self.risk_model.decompose_risk(pw)
                rr_sorted = rr.sort_values('vol_contri', ascending=False, key=abs)
                self._last_risk_report = rr_sorted
            else:
                rr_sorted = self._last_risk_report

            if top_x is not None and top_x < len(rr_sorted):
                factors = rr_sorted.index[:top_x]
                factors = [f for f in factors if f != "Specific"]
            else:
                factors = [f for f in rr_sorted.index if f != "Specific"]
        else:
            factors = factor_list

        # 2) Build the sliders
        self._scenario_sliders = {}
        slider_items = []
        for fct in factors:
            slider = widgets.FloatSlider(
                value=0.0, min=-5.0, max=5.0, step=0.1,
                description="", continuous_update=False,
                orientation='horizontal',
                layout=Layout(width='220px')
            )
            self._scenario_sliders[fct] = slider
            label = widgets.Label(f"{fct}: ", layout=Layout(width='100px'))
            row = HBox([label, slider], layout=Layout(margin='2px 0px 2px 0px'))
            slider_items.append(row)

            slider.observe(self._update_scenario_plot, names='value')

        self._sliders_vbox = VBox(slider_items, layout=Layout(height='auto'))

        self._scenario_output = widgets.Output(layout=Layout(height='400px', width='700px'))

        self._scenario_hbox = HBox([self._sliders_vbox, self._scenario_output],
                                   layout=Layout(align_items="flex-start", height='auto'))

        display(self._scenario_hbox)
        # Force an initial update
        self._update_scenario_plot()

    def _update_scenario_plot(self, change=None):
        """
        Called whenever a slider changes. We compute scenario impact & redraw a single plot
        in self._scenario_output. We do a bar chart of factor stdevs horizontally.
        """
        factor_shocks = {}
        for fct, slider in self._scenario_sliders.items():
            factor_shocks[fct] = slider.value

        scenario_ret = self.scenario_impact(factor_shocks)

        df_shocks = pd.DataFrame({
            "Factor": list(factor_shocks.keys()),
            "StDev Shock": list(factor_shocks.values())
        })
        df_shocks.sort_values("StDev Shock", key=abs, ascending=False, inplace=True)

        fig = px.bar(
            df_shocks,
            y="Factor", x="StDev Shock",
            orientation='h',
            template="plotly_white"
        )
        fig.update_layout(
            title=f"<b>Scenario Impact: {scenario_ret*100:.2f}%</b>",
            xaxis=dict(range=[-5,5])
        )
        fig.update_traces(marker_color='royalblue')

        with self._scenario_output:
            clear_output(wait=True)
            # use 'notebook' => in a classic Jupyter Notebook, 
            # this helps re-draw in the same cell output area
            fig.show("notebook")

    # ---------------------------------------------------------------------
    # 6) FACTOR EXPOSURE TREEMAP
    # ---------------------------------------------------------------------

    def plot_factor_exposure_treemap(self, portfolio_weights=None, color_abs=False):
        """
        Show a treemap of portfolio exposures aggregated by factor name,
        splitting each name by underscores for multi-level grouping.
        
        If color_abs=False, color scale is diverging with negative/positive.
        If color_abs=True, color scale is purely magnitude-based.
        """
        pw = self._check_portfolio(portfolio_weights)
        riskModelCut = self.cut_down_model(self.risk_model,pw)
        beta = riskModelCut.get_beta()  # tickers x factors
        factor_expo = pw.dot(beta)

        if factor_expo.abs().sum() < 1e-15:
            print("All factor exposures are zero; nothing to display in treemap.")
            return

        df_expo = pd.DataFrame({
            "factor": factor_expo.index,
            "exposure": factor_expo.values
        })

        # Split each factor name into multiple levels by underscore
        def split_factor_levels(name):
            parts = [p.strip() for p in name.split("_") if p.strip()]
            if not parts:
                return [name]
            return parts

        df_expo["levels"] = df_expo["factor"].apply(split_factor_levels)
        # find max depth
        max_depth = max(len(lst) for lst in df_expo["levels"])

        # create columns level_0 ... level_(max_depth-1)
        for d in range(max_depth):
            col_name = f"level_{d}"
            df_expo[col_name] = df_expo["levels"].apply(lambda x: x[d] if d < len(x) else "")

        path_cols = [f"level_{d}" for d in range(max_depth)]

        # color column
        if color_abs:
            df_expo["color"] = df_expo["exposure"].abs()
            cmin, cmax = 0.0, df_expo["color"].max()
            cscale = px.colors.sequential.Blues
            ctitle = "Abs Exposure"
        else:
            df_expo["color"] = df_expo["exposure"]
            vmax = df_expo["color"].abs().max()
            cmin, cmax = -vmax, vmax
            cscale = px.colors.diverging.RdBu
            ctitle = "Signed Exposure"

        fig = px.treemap(
            df_expo,
            path=path_cols,        # each row is one factor, with multiple levels
            values="exposure",     # numeric size
            color="color",         # color scale
            color_continuous_scale=cscale,
            range_color=(cmin, cmax),
            title="Factor Exposures Treemap" + f" - {self.name}",
        )
        fig.update_layout(coloraxis_colorbar=dict(title=ctitle))
        fig.show()

    # ---------------------------------------------------------------------
    # INTERNAL UTILITY
    # ---------------------------------------------------------------------

    def _check_portfolio(self, portfolio_weights):
        if portfolio_weights is not None:
            return portfolio_weights
        if self.portfolio is None or self.portfolio.empty:
            raise ValueError("No portfolio provided (nor set in constructor).")
        return self.portfolio

    def _plot_cumulative_factor_returns(self, factor_list, factor_rets: pd.DataFrame):
        if not factor_list:
            print("No factors to plot.")
            return
        cum_rets = (1 + factor_rets[factor_list]).cumprod() - 1

        fig = go.Figure()
        for factor in factor_list:
            fig.add_trace(go.Scatter(
                x=cum_rets.index,
                y=cum_rets[factor],
                mode='lines',
                name=factor,
                line=dict(width=2),
            ))
        fig.update_layout(
            title="Cumulative Factor Returns" + f" - {self.name}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".1%",
            legend_title="Factors",
            hovermode="x unified",
            template="plotly_white"
        )
        fig.show()

    def _simulate_forward_cone(self, 
                               portfolio_weights, 
                               hist_pnl, 
                               horizon_days, 
                               n_scenarios, 
                               quantiles):
        riskModelCut = self.cut_down_model(self.risk_model,portfolio_weights)
        raw_cov = riskModelCut.get_cov()
        factor_cov = self._nearest_posdef(raw_cov)
        beta = riskModelCut.get_beta()
        idio = riskModelCut.get_idio()

        factor_expo = portfolio_weights.dot(beta)
        L = np.linalg.cholesky(factor_cov.values)

        idio_var = (portfolio_weights**2 * idio**2).sum()
        idio_std = np.sqrt(idio_var)

        sim_paths = np.zeros((horizon_days+1, n_scenarios))
        sim_paths[0, :] = 0.0

        for d in range(1, horizon_days+1):
            z_factors = np.random.randn(n_scenarios, len(factor_cov))
            factor_shock = z_factors @ L.T
            pf_factor_ret = (factor_shock * factor_expo.values).sum(axis=1)

            z_idio = np.random.randn(n_scenarios) * idio_std
            daily_ret = pf_factor_ret + z_idio
            sim_paths[d, :] = sim_paths[d-1, :] + daily_ret

        qs_sorted = sorted(quantiles)
        out = {}
        for q in qs_sorted:
            out[f"q{q}"] = np.percentile(sim_paths, q, axis=1)

        return pd.DataFrame(out, index=range(horizon_days+1))

    def _plot_cone(self, hist_pnl: pd.Series, df_quantiles: pd.DataFrame):
        if df_quantiles.empty:
            print("No forward simulation results. Skipping cone plot.")
            return

        sorted_cols = sorted(df_quantiles.columns, key=lambda x: float(x[1:]))
        low1 = sorted_cols[0]
        high1 = sorted_cols[-1]
        low2 = sorted_cols[1] if len(sorted_cols) >= 4 else None
        high2 = sorted_cols[-2] if len(sorted_cols) >= 4 else None

        median_col = None
        if len(sorted_cols) % 2 == 1:
            mid_idx = len(sorted_cols)//2
            median_col = sorted_cols[mid_idx]

        last_val = hist_pnl.iloc[-1]
        horizon_days = len(df_quantiles) - 1
        last_date = hist_pnl.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=horizon_days,
                                     freq='B')

        df_cone = df_quantiles.copy()
        for c in df_cone.columns:
            df_cone[c] += last_val

        new_index = [last_date] + list(future_dates)
        df_cone.index = new_index

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_pnl.index, y=hist_pnl.values,
            mode='lines', name='Historical PnL', line=dict(color='black')
        ))

        fig.add_trace(go.Scatter(
            x=df_cone.index, y=df_cone[high1],
            name=f'{high1}', mode='lines',
            line=dict(color='rgba(0,0,255,0.2)')
        ))
        fig.add_trace(go.Scatter(
            x=df_cone.index, y=df_cone[low1],
            name=f'{low1}', mode='lines',
            fill='tonexty',
            line=dict(color='rgba(0,0,255,0.2)')
        ))

        if low2 and high2:
            fig.add_trace(go.Scatter(
                x=df_cone.index, y=df_cone[high2],
                name=f'{high2}', mode='lines',
                line=dict(color='rgba(0,0,255,0.4)')
            ))
            fig.add_trace(go.Scatter(
                x=df_cone.index, y=df_cone[low2],
                name=f'{low2}', mode='lines',
                fill='tonexty',
                line=dict(color='rgba(0,0,255,0.4)')
            ))

        if median_col and median_col in df_cone.columns:
            fig.add_trace(go.Scatter(
                x=df_cone.index, y=df_cone[median_col],
                name=f'{median_col}', mode='lines',
                line=dict(color='rgba(0,0,255,1)', dash='dash')
            ))

        fig.update_layout(
            title="Historical & Forward-Looking PnL Cone" + f" - {self.name}",
            xaxis_title="Date",
            yaxis_title="PnL",
            yaxis_tickformat=".1%",   #percentage
            hovermode="x unified",
            template="plotly_white"
        )
        fig.show()

    def _nearest_posdef(self, cov: pd.DataFrame, eps=1e-12) -> pd.DataFrame:
        """
        Project the covariance to the nearest positive semidefinite via eigenvalue clipping.
        Helps avoid Cholesky errors if matrix is nearly singular or not PD.
        """
        sym_cov = 0.5 * (cov + cov.T)
        vals, vecs = np.linalg.eigh(sym_cov.values)
        vals_clipped = np.clip(vals, eps, None)
        mat_pd = (vecs @ np.diag(vals_clipped) @ vecs.T)
        mat_pd = 0.5 * (mat_pd + mat_pd.T)
        return pd.DataFrame(mat_pd, index=cov.index, columns=cov.columns)
