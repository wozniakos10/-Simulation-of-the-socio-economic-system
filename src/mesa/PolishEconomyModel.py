from mesa import Model
from mesa.datacollection import DataCollector
import random
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from PolishGovernment import PolishGovernment
from PolishCentralBank import PolishCentralBank
from PolishHousehold import PolishHousehold
from PolishFirm import PolishFirm
from RLAgent import POLISH_DATA_2024


class PolishEconomyModel(Model):
    """Polish economy model with GUS 2024 data"""
    def __init__(self, N_households, N_firms):
        super().__init__()
        self.num_households = N_households
        self.step_count = 0

        self.government = PolishGovernment(self)
        self.bank = PolishCentralBank(self)
        # Creating households
        self.households = []
        for i in range(N_households):
            household = PolishHousehold(self, i)
            self.households.append(household)

        # Creating multiple firms
        self.firms = []
        for i in range(N_firms):
            firm = PolishFirm(self, firm_id=i)
            self.firms.append(firm)

        # Random assignment of workers to firms
        for h in self.households:
            assigned_firm = random.choice(self.firms)
            h.employer = assigned_firm
            assigned_firm.employees.append(h)

        for household in self.households:
            household.bank_deposits = random.randint(500, 5000)
            household.debt = 0
            household.credit_score = random.randint(580, 780)
            # Make some households immediately want loans
            if random.random() < 0.3:  # 30% want loans initially
                household.wants_loan = True

        for firm in self.firms:
            firm.bank_deposits = random.randint(2000, 20000)
            firm.debt = 0
            firm.credit_score = random.randint(620, 750)
            firm.cash_injection = 0
            # Make some firms immediately want loans
            if random.random() < 0.4:  # 40% want loans initially
                firm.wants_loan = True

        # Calculate initial total deposits
        self.bank.total_deposits = sum(h.bank_deposits for h in self.households)
        self.bank.total_deposits += sum(f.bank_deposits for f in self.firms)




        # Data collector with Polish indicators
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Household_Wealth": lambda m: sum([h.wealth for h in m.households]),
                "Average_Gross_Wage": lambda m: np.mean([h.gross_wage for h in m.households]),
                "Average_Net_Wage": lambda m: np.mean([h.net_wage for h in m.households]),
                "Budget_Revenue": lambda m: m.government.spending,
                "Total_Firm_Profit": lambda m: sum([firm.profit for firm in m.firms]),
                "Average_Firm_Profit": lambda m: np.mean([firm.profit for firm in m.firms]),
                "Total_Firm_Revenue": lambda m: sum([firm.revenue for firm in m.firms]),
                "Average_Firm_Revenue": lambda m: np.mean([firm.revenue for firm in m.firms]),
                "Budget_Balance": lambda m: m.government.budget_balance,
                "Public_Debt": lambda m: m.government.debt,
                "Debt_Ratio": lambda m: (m.government.debt / m.government.gdp_estimate) * 100,
                "Wealth_Inequality": lambda m: np.std([h.wealth for h in m.households]),
                "Average_Household_Reward": lambda m: np.mean([h.reward for h in m.households]) if m.households else 0,
                "Total_Firm_Reward": lambda m: sum([firm.reward for firm in m.firms]),
                "Average_Firm_Reward": lambda m: np.mean([firm.reward for firm in m.firms]),
                "Government_Reward": lambda m: m.government.reward,
                "Inflation_Effect": lambda m: POLISH_DATA_2024['inflation_rate'],
                "Total_Consumption": lambda m: sum([h.current_savings for h in m.households if hasattr(h, 'current_savings')]),

                # BANK DATA:
                "Bank_Deposits": lambda m: m.bank.total_deposits,
                "Bank_Loans": lambda m: m.bank.total_loans,
                "Bank_Profit": lambda m: m.bank.profit,
                "Previous_Bank_Profit": lambda m: m.bank.prev_profit,
                "Bad_Loans": lambda m: m.bank.bad_loans,
                "Bad_Loan_Ratio": lambda m: (m.bank.bad_loans / m.bank.total_loans * 100) if m.bank.total_loans > 0 else 0
            }
        )

        self.datacollector.collect(self)

    def step(self):
        self.step_count += 1

        # Update agents
        for household in self.households:
            household.step()

        # Update all firms
        for firm in self.firms:
            firm.step()

        self.government.step()
        self.bank.step()

        # Simulate business cycles (every 12 steps = 1 year)
        if self.step_count % 12 == 0:
            self.apply_economic_cycle()

        self.datacollector.collect(self)

    def apply_economic_cycle(self):
        """Simulates business cycles and economic shocks"""
        cycle_type = random.choice(['recession', 'growth', 'stabilization'])

        if cycle_type == 'recession':
            print(f"Year {self.step_count//12}: Economic recession")
            for h in self.households:
                h.wealth *= 0.95  # wealth decline
                h.gross_wage *= 0.98  # wage decline

            # Update costs for all firms
            for firm in self.firms:
                firm.base_costs *= 1.1  # higher operating costs

        elif cycle_type == 'growth':
            print(f"Year {self.step_count//12}: Economic growth")
            for h in self.households:
                h.gross_wage *= 1.03  # wage increase

            # Update costs for all firms
            for firm in self.firms:
                firm.base_costs *= 0.98  # relatively lower costs

        # Update net wages after changes
        for h in self.households:
            h.net_wage = h.calculate_net_wage(h.gross_wage)



    def visualize_polish_results(self):
        """Creates visualizations adapted to Polish data"""
        df = self.datacollector.get_model_vars_dataframe()

        # Polish font and style configuration
        plt.rcParams['font.size'] = 10
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Polish Economy Simulation with Reinforcement learning - compared to GUS 2024 Data',
                     fontsize=16, fontweight='bold')

        # 1. Macroeconomic indicators
        ax1 = plt.subplot(3, 4, 1)
        plt.plot(df.index, df['Total_Household_Wealth']/1000, 'b-', linewidth=2,
                 label='Wealth (thousand PLN)')
        print(f"Firm profits: {df['Total_Firm_Profit'].iloc[-1]}")
        print(f"Firm revenues: {df['Total_Firm_Revenue'].iloc[-1]}")
        plt.plot(df.index, df['Total_Firm_Revenue']/1000, 'g-', linewidth=2,
                 label='Firm revenue (thousand PLN)')
        plt.plot(df.index, df['Budget_Revenue']/1000, 'r-', linewidth=2,
                 label='Budget expenses (thousand PLN)')
        plt.title('Main Economic Indicators', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Value (thousand PLN)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Wages in Poland
        ax2 = plt.subplot(3, 4, 2)
        plt.plot(df.index, df['Average_Gross_Wage'], 'b-', linewidth=2, label='Gross')
        plt.plot(df.index, df['Average_Net_Wage'], 'g-', linewidth=2, label='Net')
        plt.axhline(y=POLISH_DATA_2024['average_gross_wage'], color='b', linestyle='--',
                    alpha=0.7, label='GUS 2024 Average (gross)')
        plt.axhline(y=POLISH_DATA_2024['average_net_wage'], color='g', linestyle='--',
                    alpha=0.7, label='GUS 2024 Average (net)')
        plt.title('Wages vs GUS Data', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Wage (PLN)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Public finances
        ax3 = plt.subplot(3, 4, 3)
        plt.plot(df.index, df['Budget_Balance']/1000, 'purple', linewidth=2,
                 label='Budget balance')
        plt.plot(df.index, df['Debt_Ratio'], 'red', linewidth=2,
                 label='Debt (% GDP)')
        plt.axhline(y=60, color='red', linestyle='--', alpha=0.7,
                    label='Constitutional limit (60%)')
        plt.title('Public Finances', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Social inequalities
        ax4 = plt.subplot(3, 4, 4)
        plt.plot(df.index, df['Wealth_Inequality']/1000, 'orange', linewidth=2)
        plt.title('Wealth Inequality\n(Standard deviation)', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Deviation (thousand PLN)')
        plt.grid(True, alpha=0.3)

        # 5. Government policies
        ax5 = plt.subplot(3, 4, 5)
        if hasattr(self.government, 'action_history') and len(self.government.action_history) > 0:
            gov_actions = pd.Series(self.government.action_history).value_counts()
            gov_actions.plot(kind='bar', ax=ax5, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No government actions recorded',
                     ha='center', va='center', transform=ax5.transAxes)
        plt.title('Government Strategies\n(Frequency)', fontweight='bold')
        plt.xlabel('Policy type')
        plt.ylabel('Number of periods')
        plt.grid(True, alpha=0.3)

        # 6. Firm Strategies (All Firms)
        ax6 = plt.subplot(3, 4, 6)

        # Collect all strategies from all firms
        all_strategies = []
        for firm in self.firms:
            if hasattr(firm, "action_history"):
                all_strategies.extend(firm.action_history)

        # Count the frequency of each strategy
        if all_strategies:
            strategy_counts = pd.Series(all_strategies).value_counts()
            strategy_counts.plot(kind='bar', ax=ax6, color=['skyblue', 'lightgreen', 'salmon', 'gold'])

            ax6.set_title("Firm Strategies\n(Frequency)", fontweight='bold')
            ax6.set_xlabel("Strategy Type")
            ax6.set_ylabel("Number of Periods")
            ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
            ax6.grid(True, alpha=0.3)
        else:
            ax6.axis('off')
            ax6.text(0.5, 0.5, "No firm strategy data available",
                     transform=ax6.transAxes, ha='center', va='center', fontsize=10, color='gray')
        # 7. Average household reward
        ax7 = plt.subplot(3, 4, 7)
        if 'Average_Household_Reward' in df.columns:
            plt.plot(df.index, df['Average_Household_Reward'], 'darkred', linewidth=2)
            plt.title('Average Household Reward', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Household reward data not available',
                     ha='center', va='center', transform=ax7.transAxes)
            plt.title('Household Performance', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)

        # 8. Individual agent rewards - Show government and each firm
        ax8 = plt.subplot(3, 4, 8)

        # Define a color palette
        colors = plt.cm.tab10.colors  # 10 distinct colors

        # Plot government reward
        if 'Government_Reward' in df.columns:
            ax8.plot(df.index, df['Government_Reward'],
                     color='navy', linewidth=3,
                     label='Government', marker='o', markersize=3)

        # Plot individual firm rewards if available
        firm_reward_plotted = False
        for i, firm in enumerate(self.firms):
            firm_reward_col = f'Firm_{i}_Reward'
            if firm_reward_col in df.columns:
                color = colors[i % len(colors)]  # cycle through color palette
                ax8.plot(df.index, df[firm_reward_col],
                         color=color, linewidth=2,
                         label=f'Firm {i+1}', alpha=0.8)
                firm_reward_plotted = True

        # Fallback to average if individual firm rewards not available
        if not firm_reward_plotted and 'Average_Firm_Reward' in df.columns:
            ax8.plot(df.index, df['Average_Firm_Reward'],
                     color='maroon', linewidth=2,
                     label='Firms (Average)', linestyle='--')

        ax8.set_title('Agent Performance Comparison', fontweight='bold')
        ax8.set_xlabel('Months')
        ax8.set_ylabel('Reward')
        ax8.legend(loc='best', fontsize='small')
        ax8.grid(True, alpha=0.3)

        # 9. Real Net Wage (Inflation Adjusted)
        ax9 = plt.subplot(3, 4, 9)
        inflation_rate = POLISH_DATA_2024['inflation_rate']
        time_steps = len(df['Average_Net_Wage'])

        # Cumulative inflation factor over time
        inflation_multiplier = [(1 + inflation_rate) ** t for t in range(time_steps)]

        # Calculate real wages
        real_net_wage = np.array(df['Average_Net_Wage']) / inflation_multiplier

        # Plot real vs nominal
        ax9.plot(df['Average_Net_Wage'], label='Nominal Net Wage', linestyle='--', color='gray')
        ax9.plot(real_net_wage, label='Real Net Wage', color='green')

        ax9.set_title("Real Net Wage (Inflation Adjusted)")
        ax9.set_xlabel("Time")
        ax9.set_ylabel("PLN")
        ax9.legend()
        ax9.grid(True)

        # 10. Total consumption
        ax10 = plt.subplot(3, 4, 10)
        plt.plot(df.index, df['Total_Consumption']/1000, 'forestgreen', linewidth=2)
        plt.title('Total Consumption', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Value (thousand PLN)')
        plt.grid(True, alpha=0.3)

        # 11. Cumulative Total Firm Reward
        ax11 = plt.subplot(3, 4, 11)

        if "Total_Firm_Reward" in df.columns:
            df["Cumulative_Reward"] = df["Total_Firm_Reward"].cumsum()
            ax11.plot(df.index, df["Cumulative_Reward"], color='orange', linewidth=2)
            ax11.set_title("Cumulative Firm Reward", fontweight='bold')
            ax11.set_xlabel("Months")
            ax11.set_ylabel("Cumulative Reward")
            ax11.grid(True, alpha=0.3)
        else:
            ax11.axis('off')
            ax11.text(0.5, 0.5, "Firm reward data unavailable",
                      transform=ax11.transAxes, ha='center', va='center', fontsize=10, color='gray')
        # 12. Public debt and debt ratio
        ax12 = plt.subplot(3, 4, 12)
        ax12_twin = ax12.twinx()

        # Public debt (left axis)
        line1 = ax12.plot(df.index, df['Public_Debt']/1000000, 'navy', linewidth=2,
                          label='Public debt (million PLN)')
        ax12.set_xlabel('Months')
        ax12.set_ylabel('Debt (million PLN)', color='navy')
        ax12.tick_params(axis='y', labelcolor='navy')

        # Debt ratio (right axis)
        line2 = ax12_twin.plot(df.index, df['Debt_Ratio'], 'red', linewidth=2,
                               label='Debt ratio (%)')
        ax12_twin.axhline(y=60, color='red', linestyle='--', alpha=0.7,
                          label='Constitutional limit (60%)')
        ax12_twin.set_ylabel('Debt (% GDP)', color='red')
        ax12_twin.tick_params(axis='y', labelcolor='red')

        plt.title('Public Debt Analysis', fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Layout adjustment
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Save plot
        plt.savefig('figures/polish_economy_simulation_2024.png', dpi=300, bbox_inches='tight')
        plt.show()

        self.visualize_bank_data()

        # Additional statistical report
        self._generate_statistical_report(df)

    def visualize_bank_data(self):
        """Creates charts from bank data — dynamics of loans, deposits, profits and bad loans"""
        df = self.datacollector.get_model_vars_dataframe()

        # Debug: Print available columns and sample data
        print("Available columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        print("Sample data:")
        print(df.head())

        # Check if we have any data
        if df.empty:
            print("Warning: No data collected!")
            return

        plt.rcParams['font.size'] = 10
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Bank Dynamics in Simulation — Poland 2024', fontsize=16, fontweight='bold')

        def smooth(series, window=3):
            if len(series) < window:
                return series
            return series.rolling(window=window, center=True, min_periods=1).mean()

        # Define expected column names and their alternatives
        column_mapping = {
            'Bank_Loans': ['Bank_Loans', 'total_loans', 'Total_Loans'],
            'Bank_Deposits': ['Bank_Deposits', 'total_deposits', 'Total_Deposits'],
            'Bank_Profit': ['Bank_Profit', 'profit', 'Profit'],
            'Previous_Bank_Profit': ['Previous_Bank_Profit', 'prev_profit', 'Prev_Profit'],
            'Bad_Loans': ['Bad_Loans', 'bad_loans', 'Bad_loans'],
            'Bad_Loan_Ratio': ['Bad_Loan_Ratio', 'bad_loan_ratio', 'Bad_loan_ratio']
        }

        # Find actual column names
        actual_columns = {}
        for expected, alternatives in column_mapping.items():
            found = False
            for alt in alternatives:
                if alt in df.columns:
                    actual_columns[expected] = alt
                    found = True
                    break
            if not found:
                print(f"Warning: Column {expected} not found. Available: {df.columns.tolist()}")

        # Plot 1: Bank Assets and Liabilities
        ax1 = plt.subplot(2, 2, 1)
        if 'Bank_Loans' in actual_columns and 'Bank_Deposits' in actual_columns:
            loans_data = df[actual_columns['Bank_Loans']].fillna(0)
            deposits_data = df[actual_columns['Bank_Deposits']].fillna(0)

            plt.plot(df.index, smooth(loans_data/1000), color='blue', linewidth=2, label='Loans (thousand PLN)')
            plt.plot(df.index, smooth(deposits_data/1000), color='green', linewidth=2, label='Deposits (thousand PLN)')
            plt.fill_between(df.index, smooth(loans_data/1000), alpha=0.1, color='blue')
            plt.fill_between(df.index, smooth(deposits_data/1000), alpha=0.1, color='green')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Loans/Deposits data not available', ha='center', va='center', transform=ax1.transAxes)

        plt.title('Bank Assets and Liabilities', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Value (thousand PLN)')
        plt.grid(True, alpha=0.3)

        # Plot 2: Bank Profit
        ax2 = plt.subplot(2, 2, 2)
        if 'Bank_Profit' in actual_columns:
            profit_data = df[actual_columns['Bank_Profit']].fillna(0)
            plt.plot(df.index, smooth(profit_data), 'darkgreen', linewidth=2, label='Current profit')

            if 'Previous_Bank_Profit' in actual_columns:
                prev_profit_data = df[actual_columns['Previous_Bank_Profit']].fillna(0)
                plt.plot(df.index, smooth(prev_profit_data), 'lightgreen', linewidth=2, linestyle='--', label='Previous profit')

            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Profit data not available', ha='center', va='center', transform=ax2.transAxes)

        plt.title('Bank Profit', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Profit (PLN)')
        plt.grid(True, alpha=0.3)

        # Plot 3: Bad Loans
        ax3 = plt.subplot(2, 2, 3)
        if 'Bad_Loans' in actual_columns:
            bad_loans_data = df[actual_columns['Bad_Loans']].fillna(0)
            plt.plot(df.index, smooth(bad_loans_data), 'red', linewidth=2)
            plt.fill_between(df.index, smooth(bad_loans_data), alpha=0.15, color='red')
        else:
            plt.text(0.5, 0.5, 'Bad loans data not available', ha='center', va='center', transform=ax3.transAxes)

        plt.title('Bad Loans', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Value (PLN)')
        plt.grid(True, alpha=0.3)

        # Plot 4: Bad Loan Ratio
        ax4 = plt.subplot(2, 2, 4)
        if 'Bad_Loan_Ratio' in actual_columns:
            ratio_data = df[actual_columns['Bad_Loan_Ratio']].fillna(0)
            plt.plot(df.index, smooth(ratio_data), 'darkred', linewidth=2)
            plt.fill_between(df.index, smooth(ratio_data), alpha=0.1, color='darkred')
        elif 'Bad_Loans' in actual_columns and 'Bank_Loans' in actual_columns:
            # Calculate ratio if not directly available
            bad_loans_data = df[actual_columns['Bad_Loans']].fillna(0)
            loans_data = df[actual_columns['Bank_Loans']].replace(0, 1)  # Avoid division by zero
            ratio_data = (bad_loans_data / loans_data) * 100
            plt.plot(df.index, smooth(ratio_data), 'darkred', linewidth=2)
            plt.fill_between(df.index, smooth(ratio_data), alpha=0.1, color='darkred')
        else:
            plt.text(0.5, 0.5, 'Bad loan ratio data not available', ha='center', va='center', transform=ax4.transAxes)

        plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Moderate risk (5%)')
        plt.title('Bad Loan Ratio (%)', fontweight='bold')
        plt.xlabel('Months')
        plt.ylabel('Percent (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Create figures directory if it doesn't exist
        import os
        os.makedirs('figures', exist_ok=True)

        plt.savefig('figures/bank_dynamics_poland_2024.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _generate_statistical_report(self, df):
        """Generates statistical report with comparison to GUS data"""
        print("\n" + "="*80)
        print("STATISTICAL REPORT - SIMULATION VS GUS 2024 DATA")
        print("="*80)

        # Comparison of key indicators
        final_values = df.iloc[-1]

        comparisons = {
            'Average Gross Wage': {
                'simulation': final_values.get('Average_Gross_Wage', 0),
                'gus_2024': POLISH_DATA_2024['average_gross_wage'],
                'unit': 'PLN'
            },
            'Firm Reward (ML)': {
                'simulation': final_values.get('Average_Firm_Reward', 0),
                'gus_2024': 0,  # benchmark
                'unit': 'score'
            },
            'Government Reward (ML)': {
                'simulation': final_values.get('Government_Reward', 0),
                'gus_2024': 0,  # benchmark
                'unit': 'score'
            },
            'Inflation Effect': {
                'simulation': final_values.get('Inflation_Effect', 0)*100,
                'gus_2024': POLISH_DATA_2024['inflation_rate']*100,
                'unit': '%'
            },
            'Wealth Inequality': {
                'simulation': final_values.get('Wealth_Inequality', 0),
                'gus_2024': 50000,  # approximate standard deviation
                'unit': 'PLN'
            }
        }

        for indicator, data in comparisons.items():
            difference = abs(data['simulation'] - data['gus_2024'])
            percent_difference = (difference / data['gus_2024']) * 100 if data['gus_2024'] != 0 else 0

            print(f"\n{indicator}:")
            print(f"  Simulation:    {data['simulation']:.2f} {data['unit']}")
            print(f"  GUS 2024:      {data['gus_2024']:.2f} {data['unit']}")
            print(f"  Difference:    {difference:.2f} {data['unit']} ({percent_difference:.1f}%)")

        print("\n" + "="*80)

