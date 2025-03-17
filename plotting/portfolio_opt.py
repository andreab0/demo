
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import math
from math import sqrt


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

##########################################################
# Utility: replicate_exposures_across_dates
##########################################################
def replicate_exposures_across_dates(returns_df: pd.DataFrame,
                                     exposures_df: pd.DataFrame) -> pd.DataFrame:
    all_dates = returns_df.index.unique().sort_values()
    df_list = []
    for dt in all_dates:
        tmp = exposures_df.copy()
        tmp["date"] = dt
        df_list.append(tmp)
    big_expos = pd.concat(df_list)
    big_expos.set_index("date", append=True, inplace=True)  # index=(ticker,date)
    big_expos = big_expos.reorder_levels(["date","ticker"])
    big_expos.sort_index(inplace=True)
    return big_expos

##########################################################
# Walk-forward logic in dollar terms
##########################################################
def reoptimize_walkforward_dollar(
    returns_df: pd.DataFrame,
    alpha_time_series: pd.DataFrame,
    fitter,  # e.g. factor model fitter
    exposures_df: pd.DataFrame,  # or None
    initial_risk_model,
    initial_positions: pd.Series,  # in $
    solver_func,     # (risk_model, instruments_df, curr_positions, ref_val, basket_notional) -> new_positions
    freq: int = 20,
    re_fit: bool = False) -> pd.DataFrame:
    all_dates = returns_df.index.sort_values()
    dynamic_positions = {}
    current_positions = initial_positions.copy()
    for i in range(0, len(all_dates), freq):
        sub_dates = all_dates[i : i+freq]
        if len(sub_dates) == 0:
            break
        dt_end = sub_dates[-1]
        if re_fit and fitter is not None:
            partial_returns = returns_df.loc[:dt_end].iloc[:-1] #added a 1 day trim on the returns to avoid fwd peeking
            partial_expos   = exposures_df.loc[:dt_end] if exposures_df is not None else None
            new_risk_model = fitter.fit(partial_returns, exposures=partial_expos)
        else:
            new_risk_model = initial_risk_model
        #alpha_slice = alpha_time_series.reindex(sub_dates).mean(axis=0).fillna(0.0) ##### this can introduce forward looking bias!!!!
        # rebal on sub_dates[0], use alpha up to sub_dates[0]-1 
        alpha_slice = alpha_time_series.loc[:sub_dates[0]].iloc[-1]  # e.g. last known alpha

        if new_risk_model is not None:
            instruments = new_risk_model.get_beta().copy()
        else:
            instruments = pd.DataFrame(index=returns_df.columns)
        instruments["COST"] = alpha_slice
        new_positions = solver_func(
            new_risk_model,
            instruments,
            current_positions
        )
        for dt in sub_dates:
            dynamic_positions[dt] = new_positions
        current_positions = new_positions
    positions_df = pd.DataFrame.from_dict(dynamic_positions, orient='index', columns=returns_df.columns)
    positions_df.sort_index(inplace=True)
    return positions_df

##########################################################
# Performance metrics in dollars
##########################################################
def compute_daily_pnl_dollar(
    returns_df: pd.DataFrame, 
    positions_df: pd.DataFrame,
    shift_positions_by_1: bool = True
) -> pd.Series:
    common_idx = returns_df.index.intersection(positions_df.index)
    rets = returns_df.loc[common_idx]
    pos = positions_df.loc[common_idx]
    if shift_positions_by_1:
        pos = pos.shift(1).ffill().fillna(0.0)
    daily_pnl = (pos * rets).sum(axis=1)
    return daily_pnl

def compute_wealth_from_pnl(daily_pnl: pd.Series, initial_cash: float) -> pd.Series:
    return daily_pnl.cumsum() + initial_cash

def compute_portfolio_metrics_in_dollars(
    positions_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    initial_cash: float = 1_000_000.0,
    freq: int = 252,
    shift_positions_by_1: bool = True
) -> dict:
    common_idx = positions_df.index.intersection(returns_df.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between positions_df and returns_df.")
    pos_df = positions_df.loc[common_idx]
    ret_df = returns_df.loc[common_idx]
    if shift_positions_by_1:
        pos_df = pos_df.shift(1).ffill().fillna(0.0)
    daily_pnl = (pos_df * ret_df).sum(axis=1)
    wealth = daily_pnl.cumsum() + initial_cash
    daily_ret = wealth.pct_change().fillna(0.0)
    mean_ret = daily_ret.mean()
    annual_ret = (1 + mean_ret)**freq - 1 if abs(mean_ret) < 1 else np.nan
    annual_vol = daily_ret.std() * sqrt(freq)
    sharpe = annual_ret / annual_vol if annual_vol > 1e-12 else np.nan
    neg_r = daily_ret[daily_ret < 0]
    sortino = annual_ret / (neg_r.std() * sqrt(freq)) if (len(neg_r) > 0 and neg_r.std() > 1e-12) else np.nan
    cum_curve = (1 + daily_ret).cumprod()
    roll_max = cum_curve.cummax()
    dd_series = (cum_curve / roll_max) - 1
    max_dd = dd_series.min() if not dd_series.empty else np.nan
    calmar = abs(annual_ret / max_dd) if (max_dd < 0 and not np.isnan(max_dd)) else np.nan
    return {
        "daily_pnl": daily_pnl,
        "wealth": wealth,
        "daily_returns": daily_ret,
        "AnnualReturn": annual_ret,
        "AnnualVol": annual_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": abs(max_dd) if pd.notnull(max_dd) else np.nan,
        "Calmar": calmar,
        "Skew": daily_ret.skew(),
        "Kurtosis": daily_ret.kurtosis()
    }

##########################################################
# PortfolioDashboard - fully in dollar terms
##########################################################
class PortfolioDashboard:
    def __init__(
        self,
        returns_df: pd.DataFrame,
        base_portfolio_dollars: pd.Series,
        models_dict: dict,
        static_holdings: bool = True,
        re_optimize_frequency: int = 20,
        re_fit_factor_models: bool = False,
        alpha_time_series: pd.DataFrame = None,
        solver_func = None,
        initial_cash: float = 1e7  # starting capital
    ):
        self.returns_df = returns_df
        self.base_portfolio_dollars = base_portfolio_dollars.fillna(0.0)
        self.models_dict = models_dict
        self.static_holdings = static_holdings
        self.re_optimize_frequency = re_optimize_frequency
        self.re_fit_factor_models = re_fit_factor_models
        self.alpha_time_series = alpha_time_series
        self.solver_func = solver_func
        self.initial_cash = initial_cash

        # We'll store daily $ positions for each portfolio in this dict
        self.portfolios = {}
        # We'll store performance results (daily PnL, daily returns, stats) in self.metrics_dict
        self.metrics_dict = {}
        self._prepare_portfolios()

        # Build tab placeholders
        self.tab1_out = widgets.Output()
        self.tab2_out = widgets.Output()
        self.tab3_out = widgets.Output()
        self.tab4_out = widgets.Output()
        self.tab5_out = widgets.Output()
        self.tab6_out = widgets.Output()
        # self.history_wealth_output = widgets.Output(layout=widgets.Layout(width="1000px", height="400px"))
        # self.history_table_output = widgets.Output(layout=widgets.Layout(width="1000px", height="300px"))

        self.tab1_box = widgets.VBox([self.tab1_out])
        self.tab2_box = widgets.VBox([self.tab2_out])
        self.tab3_box = widgets.VBox([self.tab3_out])
        self.tab4_box = widgets.VBox([self.tab4_out])
        self.tab5_box = widgets.VBox([self.tab5_out])
        self.tab6_box = widgets.VBox([self.tab6_out])

    def _prepare_portfolios(self):
        # 1) Non-Optimized
        nonopt_df = pd.DataFrame(
            [self.base_portfolio_dollars] * len(self.returns_df),
            index=self.returns_df.index
        )
        self.portfolios["Non-Optimized"] = nonopt_df

        # 2) Others
        for model_name, content in self.models_dict.items():
            if model_name == "Non-Optimized":
                continue
            sol = content.get("solution", {})
            risk_model = content.get("risk_model", None)
            fitter = content.get("fitter", None)
            exposures_df = content.get("exposures", None)
            if self.static_holdings:
                if not sol or "positions" not in sol:
                    logger.warning(f"{model_name} missing 'positions' => skipping.")
                    continue
                static_pos = pd.Series(sol["positions"], index=self.returns_df.columns).fillna(0.0)
                daily_df = pd.DataFrame([static_pos] * len(self.returns_df), index=self.returns_df.index)
                self.portfolios[model_name] = daily_df
            else:
                if (risk_model is None) or (self.solver_func is None) or (self.alpha_time_series is None):
                    logger.warning(f"{model_name} missing risk_model/solver/alpha => skipping dynamic.")
                    continue
                dyn_pos_df = reoptimize_walkforward_dollar(
                    returns_df=self.returns_df,
                    alpha_time_series=self.alpha_time_series,
                    fitter=fitter,
                    exposures_df=exposures_df,
                    initial_risk_model=risk_model,
                    initial_positions=self.base_portfolio_dollars,
                    solver_func=self.solver_func,
                    freq=self.re_optimize_frequency,
                    re_fit=self.re_fit_factor_models,
                )
                self.portfolios[f"{model_name} (Dynamic)"] = dyn_pos_df

        # After building self.portfolios, compute performance metrics for each
        for pname, pos_df in self.portfolios.items():
            results = compute_portfolio_metrics_in_dollars(
                positions_df=pos_df,
                returns_df=self.returns_df,
                initial_cash=self.initial_cash,
                freq=252,
                shift_positions_by_1=True
            )
            self.metrics_dict[pname] = results

    def build_dashboard(self):
        self._build_tab1_timeseries()
        self._build_tab2_distribution()
        self._build_tab3_weights()
        self._build_tab4_factor_risk()
        self._build_tab5_performance()
        self._build_tab6_history()  # New history tab
        tabs = widgets.Tab(children=[self.tab1_box, self.tab2_box, self.tab3_box, self.tab4_box, self.tab5_box, self.tab6_out])
        tabs.set_title(0, "TimeSeries")
        tabs.set_title(1, "Distribution")
        tabs.set_title(2, "Positions")
        tabs.set_title(3, "Factor Risk")
        tabs.set_title(4, "Performance")
        tabs.set_title(5, "History")
        display(tabs)

    ##################################################
    # Tab1: TimeSeries
    ##################################################
    def _build_tab1_timeseries(self):
        with self.tab1_out:
            clear_output(wait=True)
            # Increase vertical_spacing slightly to avoid title overlap.
            fig = make_subplots(
                rows=2, cols=1, 
                subplot_titles=("Daily Returns (%)", "Cumulative Returns (%)"),
                vertical_spacing=0.15
            )
            # Left-align subplot titles
            for ann in fig.layout.annotations:
                ann.x = 0.1
            color_map = px.colors.qualitative.Bold
            portfolio_list = sorted(self.portfolios.keys())
            for idx, pname in enumerate(portfolio_list):
                metrics = self.metrics_dict[pname]
                daily_ret = metrics["daily_returns"]
                cum_ret = (1 + daily_ret).cumprod() - 1
                fig.add_trace(
                    go.Scatter(
                        x=daily_ret.index,
                        y=(daily_ret * 100),
                        mode='lines',
                        name=f"{pname} (daily)",
                        line=dict(color=color_map[idx % len(color_map)])
                    ), row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=cum_ret.index,
                        y=(cum_ret * 100),
                        mode='lines',
                        name=f"{pname} (cumul)",
                        line=dict(dash='dash', color=color_map[idx % len(color_map)])
                    ), row=2, col=1
                )
            fig.update_layout(
                height=600,
                width = 1400,
                template="plotly_white",
                title="Daily & Cumulative Returns (%)"
            )
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="%", row=1, col=1)
            fig.update_yaxes(title_text="Cumulative %", row=2, col=1)
            fig.show("vscode")

    ##################################################
    # Tab2: Distribution
    ##################################################
    def _build_tab2_distribution(self):
        with self.tab2_out:
            clear_output(wait=True)
            recs = []
            for pname in sorted(self.portfolios.keys()):
                daily_ret = self.metrics_dict[pname]["daily_returns"] * 100
                for dt, val in daily_ret.items():
                    recs.append({"Date": dt, "Portfolio": pname, "ReturnPct": val})
            df_long = pd.DataFrame(recs)
            if df_long.empty:
                print("No data for distribution plot.")
                return
            fig_hist = go.Figure()
            color_map = px.colors.qualitative.Pastel
            for i, portname in enumerate(df_long["Portfolio"].unique()):
                sub = df_long[df_long["Portfolio"] == portname]
                fig_hist.add_trace(
                    go.Histogram(
                        x=sub["ReturnPct"],
                        name=portname,
                        opacity=0.4,
                        marker=dict(color=color_map[i % len(color_map)])
                    )
                )
            fig_hist.update_layout(
                title="Overlaid Distribution of Daily Returns (%)",
                barmode="overlay",
                template="plotly_white",
                width = 1400
            )
            fig_hist.update_traces(xbins=dict(size=0.2))
            fig_hist.show("vscode")
            fig_violin = px.violin(
                df_long,
                x="Portfolio",
                y="ReturnPct",
                box=True,
                points="all",
                title="Violin Plot of Daily Returns (%)",
                template="plotly_white",
                width = 1400,
            )
            fig_violin.show("vscode")

    ##################################################
    # Tab3: Positions
    ##################################################
    def _build_tab3_weights(self):
        with self.tab3_out:
            clear_output(wait=True)
            portfolio_keys = sorted(self.portfolios.keys())
            if not portfolio_keys:
                print("No portfolios found. Skipping positions tab.")
                return
            self.base_dropdown = widgets.Dropdown(
                options=portfolio_keys,
                value=portfolio_keys[0],
                description="Base (X-axis):",
                layout=widgets.Layout(width="250px")
            )
            self.bar_dropdown = widgets.Dropdown(
                options=portfolio_keys,
                value=portfolio_keys[-1] if len(portfolio_keys)>1 else portfolio_keys[0],
                description="Bar Chart:",
                layout=widgets.Layout(width="250px")
            )
            self.topn_slider = widgets.IntSlider(
                value=15,
                min=5,
                max=30,
                step=1,
                description="Top N:",
                continuous_update=False,
                layout=widgets.Layout(width="300px")
            )
            self.summary_table_area = widgets.Output()
            self.scatter_plot_area = widgets.Output()
            self.bar_plot_area = widgets.Output()
            controls_box = widgets.HBox([self.base_dropdown, self.bar_dropdown, self.topn_slider])
            top_layout = widgets.VBox([controls_box, self.summary_table_area])
            plots_box = widgets.HBox([self.scatter_plot_area, self.bar_plot_area])
            display(top_layout, plots_box)
            self._show_all_portfolio_summaries(portfolio_keys)
            def on_change(change):
                if change["type"] == "change" and change["name"] == "value":
                    self._update_weights_compare()
            self.base_dropdown.observe(on_change, names='value')
            self.bar_dropdown.observe(on_change, names='value')
            self.topn_slider.observe(on_change, names='value')
            self._update_weights_compare()

    def _show_all_portfolio_summaries(self, portfolio_keys):
        with self.summary_table_area:
            clear_output(wait=True)
            summary_rows = []
            for pname in portfolio_keys:
                if isinstance(self.portfolios[pname], pd.Series):
                    final_pos = self.portfolios[pname]
                elif isinstance(self.portfolios[pname], pd.DataFrame) and not self.portfolios[pname].empty:
                    final_pos = self.portfolios[pname].iloc[-1].fillna(0.0)
                else:
                    final_pos = pd.Series(dtype=float)
                if final_pos.empty:
                    row_data = {
                        "Portfolio": pname,
                        "Net($)": None,
                        "Gross($)": None,
                        "Min($)": None,
                        "Max($)": None,
                        "#Positions": 0,
                        "Herfindahl": None
                    }
                else:
                    net_ = final_pos.sum()
                    gross_ = final_pos.abs().sum()
                    min_ = final_pos.min()
                    max_ = final_pos.max()
                    nz = (final_pos.abs() > 1e-12).sum()
                    total_abs = final_pos.abs().sum()
                    herf = (final_pos.abs()/total_abs).pow(2).sum() if total_abs > 0 else np.nan
                    row_data = {
                        "Portfolio": pname,
                        "Net($)": net_,
                        "Gross($)": gross_,
                        "Min($)": min_,
                        "Max($)": max_,
                        "#Positions": nz,
                        "Herfindahl": herf
                    }
                summary_rows.append(row_data)
            df_all = pd.DataFrame(summary_rows).set_index("Portfolio")
            format_millions = lambda x: f"${x/1e6:.2f}M" if pd.notnull(x) else ""
            format_herf = lambda x: f"{x*100:.2f}%" if pd.notnull(x) else ""
            df_all = df_all.style.format({
                "Net($)": format_millions,
                "Gross($)": format_millions,
                "Min($)": format_millions,
                "Max($)": format_millions,
                "Herfindahl": format_herf
            })
            display(df_all)

    def _update_weights_compare(self):
        base_port = self.base_dropdown.value
        bar_port  = self.bar_dropdown.value
        top_n     = self.topn_slider.value

        # --- Scatter Plot with 45° diagonal ---
        with self.scatter_plot_area:
            clear_output(wait=True)
            if isinstance(self.portfolios[base_port], pd.Series):
                base_pos = self.portfolios[base_port]
            elif isinstance(self.portfolios[base_port], pd.DataFrame) and not self.portfolios[base_port].empty:
                base_pos = self.portfolios[base_port].iloc[-1].fillna(0.0)
            else:
                base_pos = pd.Series(dtype=float)
            if base_pos.empty:
                print(f"Base '{base_port}' final positions empty => no scatter.")
            else:
                recs = []
                for scenario_name, pos in self.portfolios.items():
                    if scenario_name == base_port:
                        continue
                    if isinstance(pos, pd.Series):
                        scen_pos = pos
                    elif isinstance(pos, pd.DataFrame) and not pos.empty:
                        scen_pos = pos.iloc[-1].fillna(0.0)
                    else:
                        scen_pos = pd.Series(dtype=float)
                    if scen_pos.empty:
                        continue
                    all_insts = base_pos.index.union(scen_pos.index)
                    for inst in all_insts:
                        recs.append({
                            "Scenario": scenario_name,
                            "Instrument": inst,
                            "Xpos": base_pos.get(inst, 0.0),
                            "Ypos": scen_pos.get(inst, 0.0)
                        })
                if recs:
                    df_scat = pd.DataFrame(recs)
                    fig_scat = px.scatter(
                        df_scat,
                        x="Xpos",
                        y="Ypos",
                        color="Scenario",
                        hover_data=["Instrument"],
                        template="plotly_white",
                        title=f"Final-Day $Positions: '{base_port}' (X) vs. Others (Y)"
                    )
                    lower = min(df_scat["Xpos"].min(), df_scat["Ypos"].min())
                    upper = max(df_scat["Xpos"].max(), df_scat["Ypos"].max())
                    fig_scat.add_trace(
                        go.Scatter(
                            x=[lower, upper],
                            y=[lower, upper],
                            mode="lines",
                            line=dict(color="black", dash="dash"),
                            name="45° Line"
                        )
                    )
                    fig_scat.update_layout(
                        xaxis_title=f"{base_port} ($)",
                        yaxis_title="Scenario ($)",
                        legend_title="Scenario",
                        width=600, height=600
                    )
                    fig_scat.show("vscode")
                else:
                    print("No other scenario has non-empty final positions => no scatter.")

        # --- Bar Chart for Top-N with Others text ---
        with self.bar_plot_area:
            clear_output(wait=True)
            if isinstance(self.portfolios[bar_port], pd.Series):
                bar_pos = self.portfolios[bar_port]
            elif isinstance(self.portfolios[bar_port], pd.DataFrame) and not self.portfolios[bar_port].empty:
                bar_pos = self.portfolios[bar_port].iloc[-1].fillna(0.0)
            else:
                bar_pos = pd.Series(dtype=float)
            if bar_pos.empty:
                print(f"Bar scenario '{bar_port}' final positions empty => no bar chart.")
            else:
                w_sorted = bar_pos.reindex(bar_pos.abs().sort_values(ascending=False).index)
                top_positions = w_sorted.iloc[:top_n]
                others = w_sorted.iloc[top_n:]
                total_abs = w_sorted.abs().sum()
                others_pct = (others.abs().sum() / total_abs * 100) if total_abs > 0 else 0
                df_bar = pd.DataFrame({"Instrument": top_positions.index, "Position($)": top_positions.values})
                colors = ["crimson" if v < 0 else "steelblue" for v in df_bar["Position($)"]]
                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        x=df_bar["Position($)"],
                        y=df_bar["Instrument"],
                        orientation='h',
                        marker_color=colors
                    )
                )
                fig_bar.update_layout(
                    template="plotly_white",
                    title=f"Top {top_n} - {bar_port} $ Positions",
                    xaxis_title="Position ($)",
                    yaxis_title="Instrument",
                    width=600, height=600
                )
                fig_bar.show("vscode")
                display(HTML(f"<div style='text-align:center; font-weight:bold;'>Others: {others_pct:.2f}% of total absolute notional</div>"))

    ##################################################
    # Tab4: Factor Risk with Selector and Historical Factor Returns
    ##################################################
    def _build_tab4_factor_risk(self):
        with self.tab4_out:
            clear_output(wait=True)
            # Build dropdown listing only models with a risk_model
            risk_model_options = [name for name, content in self.models_dict.items() if content.get("risk_model") is not None]
            if not risk_model_options:
                print("No risk models available.")
                return
            self.risk_model_dropdown = widgets.Dropdown(
                options=risk_model_options,
                description="Select Risk Model:",
                layout=widgets.Layout(width="300px")
            )
            self.fctrisk_plot_area = widgets.Output()
            display(widgets.VBox([self.risk_model_dropdown, self.fctrisk_plot_area]))
            self.risk_model_dropdown.observe(self._update_factor_risk_selected, names='value')
            # Force an initial update
            self._update_factor_risk_selected()

    def _update_factor_risk_selected(self, change=None):
        with self.fctrisk_plot_area:
            clear_output(wait=True)
            selected = self.risk_model_dropdown.value
            content = self.models_dict.get(selected, {})
            rm = content.get("risk_model")
            if rm is None:
                print(f"Risk model not available for {selected}.")
                return
            # Get the portfolio positions for this model:
            if selected in self.portfolios:
                df_positions = self.portfolios[selected]
            elif f"{selected} (Dynamic)" in self.portfolios:
                df_positions = self.portfolios[f"{selected} (Dynamic)"]
            else:
                print(f"No portfolio found for {selected}.")
                return
            if df_positions.empty:
                print(f"{selected}: positions are empty.")
                return
            final_pos = df_positions.iloc[-1].fillna(0.0)
            # Compute risk decomposition
            risk_df = rm.decompose_risk(final_pos)
            if risk_df.empty:
                print(f"{selected}: risk decomposition is empty.")
                return
            col_candidates = [c for c in risk_df.columns if "vol_contri" in c.lower()]
            if not col_candidates:
                print(f"{selected}: no 'vol_contri' column found.")
                return
            vc_col = col_candidates[0]
            # Build exposures chart: sort risk_df by absolute vol_contri and take top 10
            df_exp = risk_df.sort_values("vol_contri", key=abs, ascending=False).head(10)
            df_exp["plot_val"] = (df_exp[vc_col] / self.initial_cash) * 100
            fig_exp = go.Figure()
            fig_exp.add_trace(
                go.Bar(
                    x=df_exp.index.astype(str),
                    y=df_exp["plot_val"],
                    marker_color=["red" if idx.lower()=="specific" else "steelblue" for idx in df_exp.index]
                )
            )
            fig_exp.update_layout(
                title=f"Current Factor Exposures - {selected}",
                xaxis_title="Factor",
                yaxis_title="Exposure (% of Initial Capital)",
                template="plotly_white",
                width = 1400
                
            )
            # Historical Factor Returns:
            factor_rets = rm.get_factor_returns()
            if factor_rets.empty:
                fig_ret = go.Figure()
                fig_ret.add_annotation(text="No historical factor returns available",
                                       xref="paper", yref="paper", showarrow=False, x=0.5, y=0.5)
            else:
                # Select top factors from df_exp that are present in factor_rets
                top_factors = [f for f in df_exp.index if f in factor_rets.columns]
                if not top_factors:
                    fig_ret = go.Figure()
                    fig_ret.add_annotation(text="No matching historical data for top factors",
                                           xref="paper", yref="paper", showarrow=False, x=0.5, y=0.5)
                else:
                    # If average absolute return > 1, assume returns are percentage–points and divide by 100
                    if factor_rets[top_factors].abs().mean().max() > 1:
                        factor_rets_scaled = factor_rets[top_factors] / 100.0
                    else:
                        factor_rets_scaled = factor_rets[top_factors]
                    # Clip extreme values to reduce spikiness (adjust thresholds as needed)
                    factor_rets_scaled = factor_rets_scaled.clip(lower=-0.5, upper=0.5)
                    cum_factor_rets = (1 + factor_rets_scaled).cumprod() - 1
                    fig_ret = go.Figure()
                    for factor in top_factors:
                        fig_ret.add_trace(
                            go.Scatter(
                                x=cum_factor_rets.index,
                                y=cum_factor_rets[factor],
                                mode="lines",
                                name=factor,
                                line=dict(width=2)
                            )
                        )
                    fig_ret.update_layout(
                        title=f"Historical Cumulative Factor Returns - {selected}",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return",
                        yaxis_tickformat=".1%",
                        template="plotly_white",
                        width = 1400
                    )
            # Show the two figures sequentially inside the output widget
            fig_exp.show("vscode")
            fig_ret.show("vscode")


    ##################################################
    # Tab5: Performance
    ##################################################
    def _build_tab5_performance(self):
        with self.tab5_out:
            clear_output(wait=True)
            results = []
            for pname in sorted(self.metrics_dict.keys()):
                m = self.metrics_dict[pname]
                row = {
                    "Portfolio": pname,
                    "AnnualReturn (%)": m["AnnualReturn"] * 100 if pd.notnull(m["AnnualReturn"]) else np.nan,
                    "AnnualVol (%)": m["AnnualVol"] * 100 if pd.notnull(m["AnnualVol"]) else np.nan,
                    "Sharpe": m["Sharpe"],
                    "Sortino": m["Sortino"],
                    "MaxDD (%)": m["MaxDD"] * 100 if pd.notnull(m["MaxDD"]) else np.nan,
                    "Calmar": m["Calmar"],
                    "Skew": m["Skew"],
                    "Kurtosis": m["Kurtosis"]
                }
                results.append(row)
            df_perf = pd.DataFrame(results).set_index("Portfolio")
            display(df_perf.round(3))
            if not df_perf.empty and "AnnualReturn (%)" in df_perf.columns:
                df_bar = df_perf.reset_index()[["Portfolio","AnnualReturn (%)"]]
                fig_bar = px.bar(
                    df_bar,
                    x="Portfolio",
                    y="AnnualReturn (%)",
                    title="Annual Return (%) Comparison",
                    text="AnnualReturn (%)",
                    template="plotly_white"
                )
                fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                fig_bar.show("vscode")

    ##################################################
    # Tab6: Historical Analysis – Combined View Using DataFrames
    ##################################################
    def _build_tab6_history(self):
        with self.tab6_out:
            clear_output(wait=True)
            # Use the full date range from returns_df for the slider.
            all_dates = list(self.returns_df.index)
            self.history_date_slider = widgets.SelectionSlider(
                options=all_dates,
                value=all_dates[-1],
                description="Date:",
                continuous_update=False,
                orientation="horizontal",
                layout=widgets.Layout(width="800px")
            )
            # Determine rebalance dates (when positions change significantly)
            rebalance_dates = set()
            for pos_df in self.portfolios.values():
                if not pos_df.empty:
                    diff = pos_df.diff().abs().sum(axis=1)
                    rebalance_dates.update(pos_df.index[diff > 1e-3])
            self.rebalance_dates = sorted(rebalance_dates)
            # Create arrow buttons for jumping between rebalance dates.
            self.prev_button = widgets.Button(
                description="< Prev Rebalance", layout=widgets.Layout(width="150px")
            )
            self.next_button = widgets.Button(
                description="Next Rebalance >", layout=widgets.Layout(width="150px")
            )
            arrow_box = widgets.HBox([self.prev_button, self.next_button])
            # Assemble controls.
            controls = widgets.VBox([self.history_date_slider, arrow_box])
            # Create output areas with fixed, wider widths.
            self.history_wealth_output = widgets.Output(layout=widgets.Layout(width="1600px", height="400px"))
            self.history_perf_output = widgets.Output(layout=widgets.Layout(width="1600px", height="300px"))
            self.history_positions_output = widgets.Output(layout=widgets.Layout(width="1600px", height="400px"))
            main_box = widgets.VBox([controls, self.history_wealth_output, self.history_perf_output, self.history_positions_output])
            main_box.layout.width = "1600px"
            main_box.layout.align_items = "center"
            display(main_box)
            # Set observers.
            self.history_date_slider.observe(self._update_history, names="value")
            self.prev_button.on_click(lambda b: self._jump_rebalance(-1))
            self.next_button.on_click(lambda b: self._jump_rebalance(1))
            self._update_history()

    def _jump_rebalance(self, direction):
        current_date = self.history_date_slider.value
        dates = self.rebalance_dates
        if not dates:
            return
        if direction < 0:
            prev_dates = [d for d in dates if d < current_date]
            new_date = prev_dates[-1] if prev_dates else dates[0]
        else:
            next_dates = [d for d in dates if d > current_date]
            new_date = next_dates[0] if next_dates else dates[-1]
        self.history_date_slider.value = new_date

    def _update_history(self, change=None):
        sel_date = self.history_date_slider.value
        model_names = sorted(self.portfolios.keys())
        threshold = 1e-2  # Only consider positions above this threshold.
        perf_list = []    # List of performance metric dicts for each model.
        pos_dict = {}     # For each model, store (current positions, change from previous).
        wealth_dict = {}  # Cumulative wealth series per model.
        for name in model_names:
            pos_df = self.portfolios[name]
            if pos_df.empty:
                continue
            # Reindex and forward-fill positions over the full returns index.
            full_pos = pos_df.reindex(self.returns_df.index).ffill()
            subset = full_pos.loc[:sel_date]
            if subset.empty:
                continue
            current_pos = subset.iloc[-1].fillna(0.0)
            if len(subset) > 1:
                prev_pos = subset.iloc[-2].fillna(0.0)
                change_series = current_pos - prev_pos
            else:
                change_series = pd.Series(np.nan, index=current_pos.index)
            pos_dict[name] = (current_pos, change_series)
            nonzero = current_pos.abs() >= threshold
            num_names = nonzero.sum()
            num_long = (current_pos[nonzero] > 0).sum()
            num_short = (current_pos[nonzero] < 0).sum()
            try:
                met = compute_portfolio_metrics_in_dollars(
                    subset, self.returns_df.loc[:sel_date],
                    initial_cash=self.initial_cash, freq=252, shift_positions_by_1=True
                )
            except Exception as e:
                met = {}
            running_pnl = met["wealth"].iloc[-1] - self.initial_cash if "wealth" in met else np.nan
            perf_list.append({
                "Model": name,
                "Net ($)": current_pos.sum(),
                "Gross ($)": current_pos.abs().sum(),
                "Long ($)": current_pos.clip(lower=0).sum(),
                "Short ($)": (-current_pos.clip(upper=0)).sum(),
                "# Names": num_names,
                "# Long": num_long,
                "# Short": num_short,
                "Running PnL ($)": running_pnl,
                "Max DD (%)": met.get("MaxDD", np.nan) * 100 if met.get("MaxDD") is not None else np.nan,
                "Sharpe": met.get("Sharpe", np.nan),
                "Sortino": met.get("Sortino", np.nan)
            })
            wealth_dict[name] = met.get("wealth", None)
        # Build cumulative wealth chart.
        with self.history_wealth_output:
            clear_output(wait=True)
            fig_wealth = go.Figure()
            for name, wealth in wealth_dict.items():
                if wealth is not None:
                    fig_wealth.add_trace(go.Scatter(
                        x=wealth.index,
                        y=wealth,
                        mode="lines",
                        name=name
                    ))
            fig_wealth.update_layout(
                title="Cumulative Wealth from Inception",
                xaxis_title="Date",
                yaxis_title="Wealth ($)",
                template="plotly_white",
                width=1400,
                height=400
            )
            fig_wealth.show("vscode")
        # Build performance summary table as a DataFrame.
        with self.history_perf_output:
            clear_output(wait=True)
            if not perf_list:
                print("No performance metrics available for the selected date.")
            else:
                perf_df = pd.DataFrame(perf_list).set_index("Model")
                # Convert monetary columns to millions.
                for col in ["Net ($)", "Gross ($)", "Long ($)", "Short ($)", "Running PnL ($)"]:
                    if col in perf_df.columns:
                        perf_df[col] = (perf_df[col] / 1e6).round(2)
                for col in ["Max DD (%)", "Sharpe", "Sortino"]:
                    if col in perf_df.columns:
                        perf_df[col] = perf_df[col].round(2)
                # Here, we convert to HTML without hiding the index so that model names appear.
                html_perf = f"<div style='width:1400px;'>{perf_df.to_html()}</div>"
                from IPython.display import HTML
                display(HTML(html_perf))
        # Build combined positions table.
        with self.history_positions_output:
            clear_output(wait=True)
            all_tickers = set()
            for (cur, _) in pos_dict.values():
                tickers = cur.index[cur.abs() >= threshold]
                all_tickers.update(tickers)
            all_tickers = sorted(all_tickers)
            combined = pd.DataFrame(index=all_tickers)
            for name, (cur, chg) in pos_dict.items():
                df_temp = pd.DataFrame({
                    f"{name} Position ($)": cur.reindex(all_tickers),
                    f"{name} Change ($)": chg.reindex(all_tickers)
                })
                combined = pd.concat([combined, df_temp], axis=1)
            for col in combined.columns:
                combined[col] = (combined[col] / 1e6).round(2)
            combined = combined.dropna(how="all")
            # Do not reset the index so tickers remain as the index.
            html_pos = f"<div style='width:1400px;'>{combined.to_html()}</div>"
            display(HTML(html_pos))
