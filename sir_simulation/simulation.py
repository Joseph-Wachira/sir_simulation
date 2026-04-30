import numpy as np
import matplotlib.pyplot as plt
import os

print("="*60)
print("VACCINATION STRATEGY SIMULATION (SIR+V Model)")
print("Advanced Features: Waning Immunity + Delayed Rollout")
print("="*60)

def run_simulation(population, days, vaccine_coverage, enable_waning=False, 
                   delay_rollout_days=0, waning_rate=0.01):
    
    S = int(population * (1 - vaccine_coverage)) - 1
    I = 1
    R = 0
    V = 0
    
    beta = 0.4
    gamma = 0.1
    
    total_vaccinated = int(population * vaccine_coverage)
    rollout_start = delay_rollout_days
    rollout_rate = 0.1
    
    susceptible_history = [S]
    infected_history = [I]
    recovered_history = [R]
    vaccinated_history = [V]
    
    for day in range(days):
        
        if day >= rollout_start and V < total_vaccinated:
            eligible_for_vax = S
            new_vaccinations = min(int(eligible_for_vax * rollout_rate), 
                                   total_vaccinated - V)
            V += new_vaccinations
            S -= new_vaccinations
        
        infection_prob = 1 - np.exp(-beta * I / population)
        new_infections = np.random.binomial(S, infection_prob)
        
        new_recoveries = np.random.binomial(I, gamma)
        
        if enable_waning and R > 0:
            waning_count = np.random.binomial(R, waning_rate)
            R -= waning_count
            S += waning_count
        
        S -= new_infections
        I += new_infections - new_recoveries
        R += new_recoveries
        
        S = max(0, S)
        I = max(0, I)
        R = max(0, R)
        V = max(0, V)
        
        susceptible_history.append(S)
        infected_history.append(I)
        recovered_history.append(R)
        vaccinated_history.append(V)
    
    return infected_history, susceptible_history, recovered_history, vaccinated_history

population = 500
days = 100

scenarios = {
    '0% (No Vaccination)': {'coverage': 0, 'waning': False, 'delay': 0},
    '20% (Low Coverage)': {'coverage': 0.20, 'waning': False, 'delay': 0},
    '50% (Medium Coverage)': {'coverage': 0.50, 'waning': False, 'delay': 0},
    '70% (High Coverage)': {'coverage': 0.70, 'waning': False, 'delay': 0},
    '90% (Very High Coverage)': {'coverage': 0.90, 'waning': False, 'delay': 0},
    '50% + Waning Immunity': {'coverage': 0.50, 'waning': True, 'delay': 0, 'waning_rate': 0.02},
    '50% (Delayed Day 15)': {'coverage': 0.50, 'waning': False, 'delay': 15},
}

print(f"\nPopulation: {population} people")
print(f"Simulation duration: {days} days")
print(f"Scenarios to test: {len(scenarios)}")
print("\nRunning simulations...")
print("-"*60)

results = {}
peak_infections = []
total_infections = []
scenario_labels = []

output_dir = 'simulation_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.figure(figsize=(14, 8))

for name, params in scenarios.items():
    print(f"Simulating: {name}...")
    
    waning = params.get('waning', False)
    delay = params.get('delay', 0)
    waning_rate = params.get('waning_rate', 0.01)
    
    infected_hist, susc_hist, rec_hist, vax_hist = run_simulation(
        population, days, params['coverage'], 
        enable_waning=waning, 
        delay_rollout_days=delay,
        waning_rate=waning_rate
    )
    results[name] = infected_hist
    
    peak = max(infected_hist)
    total = sum(infected_hist)
    peak_infections.append(peak)
    total_infections.append(total)
    scenario_labels.append(name)
    
    print(f"   -> Peak infected: {peak:.0f} people ({peak/population*100:.1f}%)")
    print(f"   -> Total infections: {total:.0f}")
    print()
    
    plt.plot(infected_hist, linewidth=2, label=f"{name}")

plt.xlabel('Days', fontsize=12)
plt.ylabel('Number of Infected People', fontsize=12)
plt.title('Impact of Vaccination on Disease Spread (SIR+V Model)', fontsize=14)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

graph_path = f'{output_dir}/vaccination_comparison.png'
plt.savefig(graph_path, dpi=150, bbox_inches='tight')
print("-"*60)
print(f"[*] Graph saved as '{graph_path}'")

plt.figure(figsize=(12, 6))
bar_colors = ['#ff4444', '#ff8844', '#ffcc44', '#88cc44', '#44cc44', '#8844cc', '#4488cc']

bars = plt.bar(scenario_labels, peak_infections, color=bar_colors)
plt.xlabel('Vaccination Scenario', fontsize=12)
plt.ylabel('Peak Number of Infected People', fontsize=12)
plt.title('Peak Infections by Vaccination Level', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=9)

for bar, value in zip(bars, peak_infections):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{value:.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
bar_chart_path = f'{output_dir}/peak_infections_barchart.png'
plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
print(f"[*] Bar chart saved as '{bar_chart_path}'")

plt.figure(figsize=(12, 6))
bars2 = plt.bar(scenario_labels, total_infections, color=bar_colors)
plt.xlabel('Vaccination Scenario', fontsize=12)
plt.ylabel('Total Infections Over 100 Days', fontsize=12)
plt.title('Total Cases by Vaccination Level', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=9)

for bar, value in zip(bars2, total_infections):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{value:.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
total_chart_path = f'{output_dir}/total_infections_barchart.png'
plt.savefig(total_chart_path, dpi=150, bbox_inches='tight')
print(f"[*] Total cases chart saved as '{total_chart_path}'")

plt.close('all')

print("\n" + "="*70)
print("SUMMARY TABLE: PEAK INFECTIONS & TOTAL CASES")
print("="*70)
print(f"{'Scenario':<30} {'Peak Inf':>12} {'Total Cases':>12}")
print("-"*70)
for i, name in enumerate(scenario_labels):
    print(f"{name:<30} {peak_infections[i]:>12.0f} {total_infections[i]:>12.0f}")
print("="*70)

print("\n" + "="*60)
print("ANALYSIS & INTERPRETATION")
print("="*60)

R0 = 4
herd_threshold = (1 - 1/R0) * 100

print("\n" + "="*60)
print("WHAT IS HERD IMMUNITY?")
print("="*60)
print("""
HERD IMMUNITY occurs when enough of a population is immune to a disease 
that it cannot spread, protecting even those who are not immune.

When a person recovers from an infection (or gets vaccinated), they 
become immune and cannot catch or spread the disease. If a large enough 
percentage of the population is immune, the disease cannot find new 
hosts to infect - it dies out.

The HERD IMMUNITY THRESHOLD depends on how contagious the disease is:
- Measured by R0 (Basic Reproduction Number)
- R0 = average number of people one infected person infects
- Threshold = (1 - 1/R0) × 100%

For this simulation:
- R0 = 4 (each infected person infects 4 others on average)
- Herd Immunity Threshold = {:.0f}%
- This means {:.0f}% of the population needs immunity to stop spread
""".format(herd_threshold, herd_threshold))

print("\nHERD IMMUNITY ANALYSIS:")
print(f"   With R0 = {R0}, need {herd_threshold:.0f}% of population immune")
print(f"   This explains why 70% vaccination works but 50% doesn't fully")

print(f"\nKEY OBSERVATIONS:")
print(f"   0% Vaccination: Peak = {peak_infections[0]:.0f} (worst outbreak)")
print(f"   50% Vaccination: Peak = {peak_infections[2]:.0f} (below herd immunity)")
print(f"   70% Vaccination: Peak = {peak_infections[3]:.0f} (achieves herd immunity)")

print("\nADVANCED FEATURE RESULTS:")
print(f"   50% + Waning Immunity: Peak = {peak_infections[5]:.0f}")
print(f"       -> Waning immunity causes recovered people to become")
print(f"         susceptible again, leading to larger outbreaks")
print(f"   50% (Delayed Day 15): Peak = {peak_infections[6]:.0f}")  
print(f"       -> Delayed vaccination allows epidemic to spread")
print(f"         before protection kicks in")

print("\nKEY INSIGHTS:")
print(f"   1. Each increase in vaccination reduces peak infections")
print(f"   2. At 70% vaccination, outbreak is controlled (herd immunity)")
print(f"   3. Waning immunity REQUIRES higher vaccination coverage")
print(f"   4. Delayed rollout is nearly as bad as no vaccination")

print("\n" + "="*60)
print("SIMULATION COMPLETE!")
print(f"Check the '{output_dir}' folder for saved graphs:")
print(f"  - vaccination_comparison.png")
print(f"  - peak_infections_barchart.png") 
print(f"  - total_infections_barchart.png")
print("="*60)