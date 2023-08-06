Summary of the Recommendation Models
====================================

NewsRecLib integrates, to date, 13 recommendation models, partitioned
into two classes: *general recommenders (GeneralRec)* and
*fairness-aware recommenders (FairRec)*.

.. py:module:: newsreclib.models

All the recommendation models inherit from a common abstract class:

.. autosummary::
   abstract_recommender.AbstractRecommender

* **GeneralRec**

.. autosummary::
   general_rec.caum_module.CaumModule
   general_rec.cen_news_rec_module.CenNewsRecModule
   general_rec.dkn_module.DKNModule
   general_rec.lstur_module.LSTURModule
   general_rec.miner_module.MINERModule
   general_rec.mins_module.MINSModule
   general_rec.naml_module.NAMLModule
   general_rec.npa_module.NPAModule
   general_rec.nrms_module.NRMSModule
   general_rec.tanr_module.TANRModule

* **FairRec**

.. autosummary::
   fair_rec.manner_cr_module.CRModule
   fair_rec.manner_a_module.AModule
   fair_rec.manner_module.MANNERModule
   fair_rec.senti_debias_module.SentiDebiasModule
   fair_rec.sentirec_module.SentiRecModule
