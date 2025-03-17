import regimes.helpers.windowing as window_helper
from regimes.utils import log_change, noise, arithmetic_change
import numpy as _np, pandas as _pd
import dill
from cached_data.utils import flexCachedir
import os
import cufflinks as cf
import plotly.express as px
cf.go_offline()
cf.set_config_file(offline=False, world_readable = True)
        
class RegimeAnalysis():
    def __init__(self,dataset):
        self.data = dataset.copy()
        self.emd_kmeans = None
        self.ChangeDf, self.sec_cols, self.ret_cols = None, None, None
        self.window_size = 21
        self.window_step = 5
        self.returnPeriods = 5
        self.num_of_clusters = 2
        self.epsilon = 1e-5
        self.tol = 1e-5
        self.max_iteration = 100
        self.cluster_event_window = 45

        
    def _calcChange(self):
        self.ChangeDf = arithmetic_change(self.data,periods=self.returnPeriods,dropNA=False,concat=True).reset_index()

        self.sec_cols, self.ret_cols = [id for id in self.ChangeDf if ('D_chg' not in id) and (id != 'date')], [id for id in self.ChangeDf if 'D_chg' in id]

    def _setupEMDKmeans(self, norm_order=1):
        return 0
        
    def fit(self):
        self._calcChange()
        self.emd_kmeans = dill.load(open(os.path.join(flexCachedir(),'dummy_emd_kmeans.pkl'), 'rb'))
        ##dummy, not presenting the full algorithm for Clustering, which in this example adopts Earth Mover's Distance with K-means

    def clusterResults(self):
        x= _pd.DataFrame([k[-1] for k in window_helper.iter_window_values( window_size= self.window_size, window_step= self.window_step,values=self.ChangeDf['date'])],columns=['date'])
        ClusterID = f'ret{self.returnPeriods}size{self.window_size}step{self.window_step}'
        x[ClusterID] = dill.load(open(os.path.join(flexCachedir(),'dummy_emd_kmeans_labels.pkl'), 'rb'))
        ##dummy, not presenting the full algorithm for Clustering, which in this example adopts Earth Mover's Distance with K-means
        return x.set_index('date')
        
    def plotClustering(self):
        df = _pd.DataFrame([k[-1] for k in window_helper.iter_window_values( window_size= self.window_size, window_step= self.window_step,values=self.ChangeDf)],columns=self.ChangeDf.columns)
        df['labels'] = dill.load(open(os.path.join(flexCachedir(),'dummy_emd_kmeans_labels.pkl'), 'rb'))
        ##dummy, not presenting the full algorithm for Clustering, which in this example adopts Earth Mover's Distance with K-means

        for c in set(df.columns) - set({'date'}):
            for clust in df.labels.unique():
                df[f'{c}_cluster{clust}'] = _np.where(df.labels == clust,df[c], None)

        prices = [f'{c}_cluster{clust}' for clust in df.labels.unique() for c in self.sec_cols]
        
        palette = ['rgb(77,175,74)',             
                   'rgb(55,126,184)',
                    'rgb(228,26,28)',
                    'rgb(152,78,163)',
                    'rgb(255,127,0)',
                    'rgb(255,255,51)',
                    'rgb(166,86,40)',
                    'rgb(247,129,191)',
                    'rgb(153,153,153)']
        colors = [palette[clust % len(palette)] for clust in df.labels.unique() for _ in self.sec_cols]

        fig = df.iplot(x='date', y=prices, colors=colors,asFigure=True)

        ####### DETECT LABEL CHANGE DATES #######
        df['cluster_change'] = (df['labels'] != df['labels'].shift(1)).astype(int)
        change_dates = df.loc[df['cluster_change'] == 1, 'date'].dropna().tolist()

        # We only allow events that are within [cluster_event_window] days from a label change date
        min_date, max_date = df['date'].min(), df['date'].max()

        # For each event, check if it's near any cluster-change date
        for e in eventsCatalog:
            event_date = _pd.to_datetime(e['date'])
            if len(change_dates) == 0:
                continue  # no cluster changes, skip

            # find the closest cluster change date
            closest_change_date = min(change_dates, key=lambda x: abs(x - event_date))
            day_diff = abs(closest_change_date - event_date).days

            # only add if within the window and within the dataset range
            if (day_diff <= self.cluster_event_window) and (min_date <= event_date <= max_date):
                fig.add_vline(
                    x=event_date,
                    line_width=1,
                    line_dash="dash",
                    line_color="gray"
                )
                fig.add_annotation(
                    x=event_date,
                    y=0.99,
                    xref="x",
                    yref="paper",
                    text=e["label"],
                    showarrow=False,
                    font=dict(size=9),
                    textangle=90,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=0.2,
                    align="left"
                )        
        # Adjust margins or layout so annotations have room
        fig.update_layout(
            margin=dict(l=50, r=50, t=100, b=100),
            width=1000,  # you can tweak to your liking
            height=600,
        )

        fig.show()




eventsCatalog = [
    # 2000-2002: Dotcom Bubble & Aftermath
    {"date": "2000-03-10", "label": "Dotcom Bubble Peak (NASDAQ hits all-time high)"},
    {"date": "2001-09-11", "label": "9/11 Attacks in the US"},
    {"date": "2002-07-01", "label": "WorldCom Scandal & Corporate Accounting Crisis"},
    
    # 2003-2007: Pre-Subprime
    {"date": "2003-03-20", "label": "US-led Invasion of Iraq"},
    
        # 2003–2006: Fed hikes & economic recovery
    {"date": "2003-06-25", "label": "Fed Cuts Rates to 1% (Lowest in 45 years)"},
    #{"date": "2004-06-30", "label": "Fed Begins 'Measured Pace' Hike Cycle"},
    {"date": "2005-12-13", "label": "Fed Funds Rate Raised to 4.25%"},
    {"date": "2006-06-29", "label": "Fed Funds Rate Peaked at 5.25%"},
    
    {"date": "2007-07-01", "label": "Start of Subprime Meltdown"},
    
    # 2008-2009: Global Financial Crisis (GFC)
    {"date": "2008-03-16", "label": "Bear Stearns Collapse"},
    {"date": "2008-09-15", "label": "Lehman Brothers Bankruptcy"},
    {"date": "2009-03-09", "label": "Market Bottom During GFC"},
    
    # 2010-2012: Eurozone Debt Crisis
    {"date": "2010-04-23", "label": "Greece Requests Bailout (Eurozone Crisis)"},
    {"date": "2011-08-05", "label": "US Credit Rating Downgrade by S&P"},
    
    # 2012–2014: Eurozone crisis eases, taper talk
    {"date": "2012-07-26", "label": "Draghi's 'Whatever It Takes' Speech (ECB)"},
    {"date": "2012-09-13", "label": "Fed Announces QE3"},
    #{"date": "2013-05-22", "label": "Bernanke Hints at QE Tapering (Taper Tantrum)"},
    {"date": "2014-06-05", "label": "ECB Moves Deposit Rate to -0.1% (Negative)"},
    
    # 2013-2016: Various Global Shifts
    #{"date": "2015-06-29", "label": "Greek Banks Close & Capital Controls"},
    {"date": "2016-06-23", "label": "Brexit Referendum (UK)"},
    {"date": "2016-11-08", "label": "US Election (Trump Victory)"},
    
    # 2017-2019: Trade Tensions & Rate Hikes
    {"date": "2018-02-05", "label": "VIX 'Volmageddon' (Volatility Spike)"},
    {"date": "2019-05-10", "label": "US-China Trade War Escalation (Tariffs)"},
        # 2019–2020: From Fed cuts to COVID-19 shock
    {"date": "2019-07-31", "label": "Fed Cuts Rates (First Since 2008)"},
    #{"date": "2019-10-11", "label": "US-China 'Phase One' Deal Hopes"},
    {"date": "2020-01-21", "label": "First US COVID-19 Case Confirmed"},
    {"date": "2020-03-03", "label": "Fed Emergency Rate Cut (COVID-19)"},

    # 2020-2021: COVID-19 Era
    {"date": "2020-03-11", "label": "WHO Declares COVID-19 a Pandemic"},
    #{"date": "2020-03-15", "label": "Fed Slashes Rates to Near Zero"},
    {"date": "2021-09-20", "label": "Evergrande Crisis in China"},
    
    # 2022-2023: Recent Turbulence
    {"date": "2022-02-24", "label": "Russia Invades Ukraine"},
    {"date": "2022-06-15", "label": "Fed's Largest Rate Hike since 1994"},
    {"date": "2023-03-19", "label": "UBS Rescue of Credit Suisse"},
]