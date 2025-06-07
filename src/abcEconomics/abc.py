import abcEconomics as abc
import random
import pandas as pd


class Household(abc.Agent):
    def init(self):
        # Initialize with random wealth
        self.wealth = random.randint(50, 100)
        self.wage = 10

    def step(self):
        # Receive wage
        self.wealth += self.wage

        # Pay taxes
        tax = self.wage * self.government.tax_rate
        self.wealth -= tax
        self.government.revenue += tax

        # Consume
        spending = min(self.wealth, random.randint(5, 15))
        self.firm.revenue += spending
        self.wealth -= spending


class Firm(abc.Agent):
    def init(self):
        self.revenue = 0
        self.profit = 0

    def step(self):
        # Get households from the simulation
        households = self.households

        # Pay wages
        wage_bill = sum([h.wage for h in households])
        self.revenue -= wage_bill
        for h in households:
            h.wealth += h.wage

        # Pay taxes
        tax = max(self.revenue, 0) * self.government.tax_rate
        self.revenue -= tax
        self.government.revenue += tax

        # Record profit
        self.profit = self.revenue
        self.revenue = 0


class Bank(abc.Agent):
    def init(self):
        self.total_deposits = 0
        self.interest_rate = 0.01

    def step(self):
        # Get households from the simulation
        households = self.households

        # Pay interest to depositors
        for h in households:
            interest = h.wealth * self.interest_rate
            h.wealth += interest
            self.total_deposits -= interest


class Government(abc.Agent):
    def init(self):
        self.revenue = 0
        self.spending = 0
        self.tax_rate = 0.2

    def step(self):
        # Get households from the simulation
        households = self.households

        # Redistribute income
        if households:  # Check if there are households
            per_household = self.revenue / len(households)
            for h in households:
                h.wealth += per_household

        self.spending = self.revenue
        self.revenue = 0


def main():
    # Set up simulation parameters
    num_households = 10
    simulation_rounds = 50

    # Create the simulation with database disabled to avoid SQLAlchemy errors
    simulation = abc.Simulation(database_url=None)

    # Add agent types
    household = simulation.build_agents(Household, 'household', num_households)
    firm = simulation.build_agents(Firm, 'firm', 1)
    bank = simulation.build_agents(Bank, 'bank', 1)
    government = simulation.build_agents(Government, 'government', 1)

    # Store references for easier data collection
    simulation.households = household
    simulation.firm = firm[0]
    simulation.bank = bank[0]
    simulation.government = government[0]

    # Make agents accessible to each other
    for agent_group in [household, firm, bank, government]:
        for agent in agent_group:
            agent.households = household
            agent.firm = firm[0]
            agent.bank = bank[0]
            agent.government = government[0]

    # Manual data collection (since we disabled database)
    household_wealth_data = []
    firm_profit_data = []
    government_revenue_data = []
    government_spending_data = []

    # Run the simulation
    for round_num in range(simulation_rounds):
        simulation.advance_round(round_num)

        # Run steps for all agents
        household.step()
        firm.step()
        bank.step()
        government.step()

        # Collect data manually
        household_wealth_data.append({
            'round': round_num,
            'total_wealth': sum([h.wealth for h in household]),
            'households': [h.wealth for h in household]
        })

        firm_profit_data.append({
            'round': round_num,
            'profit': firm[0].profit
        })

        government_revenue_data.append({
            'round': round_num,
            'revenue': government[0].revenue,
            'spending': government[0].spending
        })

    # Create DataFrame from collected data
    household_df = pd.DataFrame(household_wealth_data)
    firm_df = pd.DataFrame(firm_profit_data)
    government_df = pd.DataFrame(government_revenue_data)

    # Print summary statistics
    print("\nHousehold Wealth Statistics:")
    print(f"Total household wealth at end: {household_df.iloc[-1]['total_wealth']}")

    print("\nFirm Statistics:")
    print(f"Final profit: {firm_df.iloc[-1]['profit']}")

    print("\nGovernment Statistics:")
    print(f"Final revenue: {government_df.iloc[-1]['revenue']}")
    print(f"Final spending: {government_df.iloc[-1]['spending']}")

    return simulation, household_df, firm_df, government_df


if __name__ == "__main__":
    simulation, household_data, firm_data, government_data = main()
    print("\nSimulation completed successfully.")