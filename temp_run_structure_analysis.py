import pandas as pd
import numpy as np
from pathlib import Path

# Quick simulation of the structure comparison
DATA_DIR = Path("data/processed")
df = pd.read_csv(DATA_DIR / "transactions.csv")
drinks_df = df[df['own_cup'].notna()].copy()

AVG_DRINK_PRICE = drinks_df['price'].mean()
AVG_DRINK_COST = drinks_df['production_cost'].mean()

print("Testing different loyalty program structures...\n")
print("="*70)

thresholds = [4, 5, 6, 7, 8, 9, 10, 12, 15]

for t in thresholds:
    discount = (1/(t+1))*100
    # Rough estimate: shorter cycles = higher response
    freq_increase = 15 + (9-t)*1.5 if t <= 9 else 15 - (t-9)*0.8
    freq_increase = max(5, min(25, freq_increase))
    
    # Quick ROI estimate
    cost_per_cycle = AVG_DRINK_COST
    margin_per_cycle = (t * (AVG_DRINK_PRICE - AVG_DRINK_COST))
    net_per_cycle = margin_per_cycle * (freq_increase/100) - cost_per_cycle
    
    # Scale to annual for 1410 members
    annual_benefit = net_per_cycle * (1410 * 10 / t)  # Rough estimate
    
    print(f"Buy {t:2d} Get 1: {discount:5.1f}% discount | {freq_increase:4.1f}% freq ↑ | ${annual_benefit:7,.0f} benefit")

print("="*70)
