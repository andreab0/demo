import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Callable, Union
import seaborn as sns
import plotly.graph_objects as go
import scipy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FactorModelValidator:
    def __init__(self, returns: pd.DataFrame, fitters: Dict[str, object], 
                 exposures_df: Optional[pd.DataFrame] = None, n_jobs: int = 1):
        self.returns = returns.sort_index()
        self.fitters = fitters
        self.exposures_df = exposures_df
        self.n_jobs = n_jobs
        self.results_ = {}
        logging.debug(f"Initialized FactorModelValidator with {len(returns)} rows of returns data.")

    def run_out_of_sample(self, train_start=None, train_end=None, 
                          test_start=None, test_end=None, train_frac=None, 
                          metric_func=None, metric_name="cs_R2"):
        logging.debug("Starting out-of-sample validation.")
        if metric_func is None:
            metric_func = self._cross_sectional_r2

        all_dates = self.returns.index.unique().sort_values()
        if train_frac is not None:
            split_idx = int(len(all_dates) * train_frac)
            train_dates = all_dates[:split_idx]
            test_dates = all_dates[split_idx:]
            train_start, train_end = train_dates[0], train_dates[-1]
            test_start, test_end = test_dates[0], test_dates[-1]
        logging.debug(f"Train period: {train_start} to {train_end}, Test period: {test_start} to {test_end}")

        in_sample = self.returns.loc[train_start:train_end]
        out_sample = self.returns.loc[test_start:test_end]

        out = {}
        for model_name, fitter in self.fitters.items():
            logging.debug(f"Fitting model '{model_name}' with {len(in_sample)} training samples.")
            fm = self._fit_model(fitter, in_sample)
            recs = self._evaluate_daily(fm, out_sample, metric_func, metric_name)
            out[model_name] = recs
            logging.debug(f"Completed evaluation for model '{model_name}'.")

        self.results_["single_split"] = out
        return out

    def _fit_model(self, fitter, in_sample_data):
        try:
            logging.debug(f"Fitting model with data shape: {in_sample_data.shape}")
            fm = fitter.fit(in_sample_data, exposures=self.exposures_df)
            logging.debug("Model fitted successfully.")
            return fm
        except Exception as e:
            logging.error(f"Error fitting model: {e}")
            raise

    def _evaluate_daily(self, fm, test_data, metric_func, metric_name):
        daily_recs = []
        for dt in test_data.index.unique():
            try:
                real_ret = test_data.loc[dt]
                pred_ret = self._predict_one_day(fm, dt, real_ret.index)

                # Align on a common index so they have matching shapes
                aligned = pd.concat([real_ret, pred_ret], axis=1, join="inner")
                # If you prefer outer join, do fillna(0) or something sensible
                # aligned = pd.concat([real_ret, pred_ret], axis=1, join="outer").fillna(0)

                # Then pass the aligned columns to your metric
                metric_value = metric_func(aligned.iloc[:, 0], aligned.iloc[:, 1])
                daily_recs.append({"date": dt, metric_name: metric_value})
            except Exception as e:
                logging.error(f"Error evaluating daily metric for {dt}: {e}")
                daily_recs.append({"date": dt, metric_name: np.nan})

        return pd.DataFrame(daily_recs).set_index("date")

    def _predict_one_day(self, factor_risk_model, date, tickers):
        try:
            factor_rets = factor_risk_model.get_factor_returns()  # shape: (factors,) for each date
            exposures = factor_risk_model.get_beta()              # shape: (tickers, factors) typically

            if factor_rets is None or exposures is None:
                # Fall back to zeros
                return pd.Series(0.0, index=tickers)

            date_str = date.strftime("%Y-%m-%d")
            latest_available_date = factor_rets.index.max()

            if date > latest_available_date:
                # use last known exposures/returns
                exposures_date = exposures.loc[tickers]  # shape: (#tickers, #factors)
                # factor_rets might also need last known row
                if date_str in factor_rets.index:
                    fr = factor_rets.loc[date_str]
                else:
                    fr = factor_rets.iloc[-1]
            else:
                if date_str in factor_rets.index:
                    fr = factor_rets.loc[date_str]
                    exposures_date = exposures.loc[tickers]
                else:
                    return pd.Series(0.0, index=tickers)

            # The key: reindex the columns (factors) of exposures_date to match fr.index
            # Because exposures_date is (#tickers, #factors) => columns = factors
            exposures_date = exposures_date.reindex(columns=fr.index, fill_value=0)

            predicted_ret = exposures_date.dot(fr)
            return predicted_ret

        except Exception as e:
            logging.error(f"Error predicting one day returns for {date_str}: {e}")
            return pd.Series(0.0, index=tickers)
    
    def _cross_sectional_r2(self, real_ret, pred_ret):
        aligned = pd.concat([real_ret, pred_ret], axis=1, join="inner").dropna()
        # aligned.columns => [0: real_ret, 1: pred_ret]
        y = aligned.iloc[:, 0].values
        yhat = aligned.iloc[:, 1].values

        if len(y) < 2:
            return np.nan
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        return r2
    
    # --------------------- Plotting Utility ---------------------

    def plot_model_performance(
        self,
        results_dict: Dict[str, pd.DataFrame],
        metric_name: str = "cs_R2",
        title: str = "Model Performance Over Time"
    ):
        """
        Plot each model's daily metric on one figure.
        """
        fig = go.Figure()
        for model_name, df_ in results_dict.items():
            if metric_name not in df_.columns:
                print(f"Warning: {metric_name} not in {model_name}. Skipping plot.")
                continue
            fig.add_trace(go.Scatter(
                x=df_.index,
                y=df_[metric_name],
                mode='lines',
                name=model_name
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=metric_name,
            hovermode="x unified",
            template="plotly_white"
        )
        fig.show()

    def plot_r2_boxplot(self,results_dict, metric_name="cs_R2"):
        # combine results for all models into a single long DataFrame
        # with columns: ['model', 'date', 'value']
        rows = []
        for model_name, df_ in results_dict.items():
            for dt, val in df_[metric_name].dropna().items():
                rows.append((model_name, dt, val))

        data = pd.DataFrame(rows, columns=["model", "date", "value"])
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data, x="model", y="value", showfliers=True)
        sns.stripplot(data=data, x="model", y="value", color="black", alpha=0.3)
        plt.title("Distribution of Daily Cross-Sectional R² by Model")
        plt.ylabel("Cross-Sectional R²")
        plt.show()

    def plot_rolling_r2(self,results_dict, metric_name="cs_R2", window=10):
        plt.figure(figsize=(10, 6))
        for model_name, df_ in results_dict.items():
            roll_series = df_[metric_name].rolling(window).mean()
            plt.plot(roll_series.index, roll_series, label=model_name)
        plt.title(f"{window}-day Rolling Average of Daily R²")
        plt.legend()
        plt.show()


    def run_walk_forward(self, train_period=None, test_period=None, 
                        metric_func: Optional[Callable[[pd.Series, pd.Series], float]] = None, 
                        metric_name: str = "cs_R2", window_step: Union[str, pd.DateOffset] = "1MS",
                        min_periods = 300, agg_func = "mean", store_predictions=True):
        """
        Enhanced walk-forward validation that can store predictions for plotting.
        
        Parameters
        ----------
        ... (previous parameters remain the same)
        store_predictions : bool, optional
            If True, stores real vs predicted values for later plotting.
        """
        logging.debug("Starting walk-forward validation.")
        if metric_func is None:
            metric_func = self._cross_sectional_r2

        all_dates = self.returns.index.unique()
        walk_forward_results = {}
        predictions_store = {}  # Store predictions if requested

        for model_name, fitter in self.fitters.items():
            logging.debug(f"Running walk-forward for model: {model_name}")
            model_wf_results = []
            model_predictions = []

            for test_end in pd.date_range(start=all_dates.min() + pd.DateOffset(months=min_periods // 20),
                                        end=all_dates.max(), freq=window_step):
                train_end = test_end - pd.DateOffset(days=1)
                train_start = train_end - pd.DateOffset(months=min_periods // 20)

                in_sample = self.returns.loc[train_start:train_end]
                out_sample = self.returns.loc[test_end:test_end + pd.DateOffset(months=1) - pd.DateOffset(days=1)]

                if len(in_sample) < min_periods:
                    logging.debug(f"Skipping period due to insufficient data (min_periods={min_periods})")
                    model_wf_results.append((test_end, np.nan))
                    continue

                fm = self._fit_model(fitter, in_sample)
                recs = self._evaluate_daily(fm, out_sample, metric_func, metric_name)
                model_wf_results.append((test_end, getattr(recs[metric_name], agg_func)()))
                
                if store_predictions:
                    # Store real and predicted values for each test day
                    for dt in out_sample.index.unique():
                        real_ret = out_sample.loc[dt]
                        pred_ret = self._predict_one_day(fm, dt, real_ret.index)
                        
                        # Align predictions with actual returns
                        aligned = pd.concat([real_ret, pred_ret], axis=1, join='inner')
                        aligned.columns = ['real', 'pred']
                        
                        model_predictions.append({
                            'date': dt,
                            'real_returns': aligned['real'],
                            'predicted_returns': aligned['pred']
                        })

            walk_forward_results[model_name] = pd.DataFrame(model_wf_results, 
                                                        columns=["test_end", metric_name]).set_index("test_end")
            
            if store_predictions:
                predictions_store[model_name] = model_predictions

        self.results_["walk_forward"] = walk_forward_results
        if store_predictions:
            self.results_["predictions"] = predictions_store
        return walk_forward_results

    def plot_real_vs_predicted(self, model_name=None, ci_level=0.95):
        """
        Plot real vs predicted returns from walk-forward validation, including confidence intervals.
        
        Parameters
        ----------
        model_name : str, optional
            Which model to plot. If None and only one model exists, uses that one.
        ci_level : float, optional
            Confidence interval level (0-1) for prediction bands.
        """
        if "predictions" not in self.results_:
            raise ValueError("No predictions found. Run walk_forward with store_predictions=True first.")
        
        predictions = self.results_["predictions"]
        if not predictions:
            raise ValueError("No predictions data available.")
        
        if model_name is None:
            if len(predictions) == 1:
                model_name = list(predictions.keys())[0]
            else:
                raise ValueError("Multiple models found. Please specify model_name.")
                
        model_preds = predictions[model_name]
        
        # Combine all predictions into a panel
        all_dates = []
        all_real = []
        all_pred = []
        
        for pred_dict in model_preds:
            date = pred_dict['date']
            real_rets = pred_dict['real_returns']
            pred_rets = pred_dict['predicted_returns']
            
            # Extend our lists with the cross-section
            all_dates.extend([date] * len(real_rets))
            all_real.extend(real_rets.values)
            all_pred.extend(pred_rets.values)
        
        df_plot = pd.DataFrame({
            'date': all_dates,
            'real': all_real,
            'predicted': all_pred
        })
        
        # Sort by date for time series plot
        df_plot = df_plot.sort_values('date')
        
        # Calculate confidence intervals if requested
        if ci_level is not None:
            # Group by date to get cross-sectional stats
            stats = df_plot.groupby('date').agg({
                'real': 'mean',
                'predicted': ['mean', 'std']
            })
            stats.columns = ['real_mean', 'pred_mean', 'pred_std']
            z_score = stats.index.map(lambda x: scipy.stats.norm.ppf((1 + ci_level) / 2))
            stats['ci_upper'] = stats['pred_mean'] + z_score * stats['pred_std']
            stats['ci_lower'] = stats['pred_mean'] - z_score * stats['pred_std']
        
        # Create plot
        fig = go.Figure()
        
        # Add real returns
        fig.add_trace(go.Scatter(
            x=stats.index,
            y=stats['real_mean'],
            mode='lines',
            name='Real Returns',
            line=dict(color='blue')
        ))
        
        # Add predicted returns
        fig.add_trace(go.Scatter(
            x=stats.index,
            y=stats['pred_mean'],
            mode='lines',
            name='Predicted Returns',
            line=dict(color='red')
        ))
        
        # Add confidence intervals if calculated
        if ci_level is not None:
            fig.add_trace(go.Scatter(
                x=stats.index,
                y=stats['ci_upper'],
                mode='lines',
                name=f'{ci_level*100}% CI Upper',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=stats.index,
                y=stats['ci_lower'],
                mode='lines',
                name=f'{ci_level*100}% CI Lower',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Real vs Predicted Returns - {model_name}",
            xaxis_title="Date",
            yaxis_title="Return",
            template="plotly_white",
            hovermode="x unified"
        )
        fig.show()

    def calculate_performance_metrics(self, real_returns, predicted_returns):
        """
        Calculate comprehensive performance metrics for factor model evaluation.
        
        Parameters
        ----------
        real_returns : pd.Series
            Actual returns
        predicted_returns : pd.Series
            Model predicted returns
            
        Returns
        -------
        dict
            Dictionary of performance metrics
        """
        metrics = {}
        
        # 1. R-squared (existing implementation)
        metrics['R2'] = self._cross_sectional_r2(real_returns, predicted_returns)
        
        # 2. Mean Absolute Error (MAE)
        metrics['MAE'] = np.mean(np.abs(real_returns - predicted_returns))
        
        # 3. Root Mean Squared Error (RMSE)
        metrics['RMSE'] = np.sqrt(np.mean((real_returns - predicted_returns)**2))
        
        # 4. Information Coefficient (Spearman rank correlation)
        metrics['IC'] = real_returns.corr(predicted_returns, method='spearman')
        
        # 5. Directional Accuracy (% of correct sign predictions)
        correct_signs = np.sign(real_returns) == np.sign(predicted_returns)
        metrics['DirectionalAccuracy'] = np.mean(correct_signs)
        
        # 6. Tracking Error (std dev of difference)
        metrics['TrackingError'] = np.std(real_returns - predicted_returns)
        
        # 7. Information Ratio
        pred_error = real_returns - predicted_returns
        metrics['InformationRatio'] = np.mean(pred_error) / np.std(pred_error) if np.std(pred_error) > 0 else np.nan
        
        return metrics

    def generate_performance_report(self, results_dict=None):
        """
        Generate a comprehensive performance report for all models.
        
        Parameters
        ----------
        results_dict : dict, optional
            Results dictionary from walk_forward validation.
            If None, uses the last stored results.
            
        Returns
        -------
        pd.DataFrame
            Performance report with all metrics for all models
        """
        if results_dict is None:
            if "walk_forward" not in self.results_:
                raise ValueError("No results found. Run walk_forward validation first.")
            results_dict = self.results_["walk_forward"]
        
        report_data = []
        
        for model_name, results in results_dict.items():
            model_metrics = {
                'Model': model_name,
                'Mean R²': results.mean().iloc[0],
                'Std R²': results.std().iloc[0],
                'Min R²': results.min().iloc[0],
                'Max R²': results.max().iloc[0],
                'Stability': 1 - (results.std().iloc[0] / abs(results.mean().iloc[0])) if results.mean().iloc[0] != 0 else np.nan
            }
            report_data.append(model_metrics)
        
        return pd.DataFrame(report_data)