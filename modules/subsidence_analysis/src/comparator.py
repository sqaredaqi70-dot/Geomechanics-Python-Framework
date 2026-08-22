"""
Subsidence Methods Comparator Module
====================================
Aggregates, scales, and compares results from different subsidence models 
(Geertsma, Gibson, DIF, Segall, Rigid Basement, etc.).

Author: Saeed Gharedaghi
License: Dual License (AGPL-3.0 / Commercial)
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class SubsidenceComparator:
    """
    Benchmarking tool to compare maximum subsidence across multiple models.
    """
    
    def __init__(self, baseline_method: str = "Geertsma"):
        """
        Parameters
        ----------
        baseline_method : str
            The method used as the reference (denominator) for ratio calculations.
        """
        self.baseline = baseline_method
        self.methods_data: Dict[str, float] = {}
        
    def add_method_result(self, method_name: str, max_subsidence_cm: float):
        """
        Add a computed max subsidence value for a specific method.
        """
        self.methods_data[method_name] = max_subsidence_cm
        
    def add_scaled_literature_result(
        self, 
        method_name: str, 
        literature_subsidence_cm: float, 
        literature_dp_mpa: float, 
        target_dp_mpa: float
    ):
        """
        Add a result from literature/other papers, scaled linearly to the target depletion.
        """
        if literature_dp_mpa <= 0:
            raise ValueError("Literature depletion must be > 0")
        
        scaled_sub = literature_subsidence_cm * (target_dp_mpa / literature_dp_mpa)
        self.methods_data[method_name] = scaled_sub
        
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Generates a standardized comparison dataframe.
        
        Returns
        -------
        pd.DataFrame
            Table with Peak_cm, Peak_inch, Ratio_to_baseline, and Delta_pct.
        """
        if not self.methods_data:
            return pd.DataFrame()
            
        base_val = self.methods_data.get(self.baseline, None)
        
        rows = []
        for name, sub_cm in self.methods_data.items():
            ratio = sub_cm / base_val if base_val and base_val > 0 else np.nan
            delta_pct = (ratio - 1.0) * 100.0 if np.isfinite(ratio) else np.nan
            
            rows.append({
                "Method": name,
                "Max_Subsidence_cm": round(sub_cm, 4),
                "Max_Subsidence_inch": round(sub_cm * 0.393701, 5),
                f"Ratio_to_{self.baseline}": round(ratio, 4),
                "Delta_pct": round(delta_pct, 2)
            })
            
        df = pd.DataFrame(rows).sort_values("Max_Subsidence_cm", ascending=False)
        return df.reset_index(drop=True)
