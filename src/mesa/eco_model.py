from mesa import Agent, Model
from mesa.datacollection import DataCollector
import random


class Household(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.wealth = random.randint(50, 100)
        self.wage = 10

    def step(self):
        # Receive wage
        self.wealth += self.wage
        # Pay taxes
        tax = self.wage * self.model.government.tax_rate
        self.wealth -= tax
        self.model.government.revenue += tax
        # Consume
        spending = min(self.wealth, random.randint(5, 15))
        self.model.firm.revenue += spending
        self.wealth -= spending


class Firm(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.revenue = 0
        self.profit = 0

    def step(self):
        # Pay wages
        wage_bill = sum([h.wage for h in self.model.households])
        self.revenue -= wage_bill
        for h in self.model.households:
            h.wealth += h.wage
        # Pay taxes
        tax = max(self.revenue, 0) * self.model.government.tax_rate
        self.revenue -= tax
        self.model.government.revenue += tax
        # Profit
        self.profit = self.revenue
        self.revenue = 0


class Bank(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.total_deposits = 0
        self.interest_rate = 0.01

    def step(self):
        # Pay interest to depositors
        for h in self.model.households:
            interest = h.wealth * self.interest_rate
            h.wealth += interest
            self.total_deposits -= interest


class Government(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.revenue = 0
        self.spending = 0
        self.tax_rate = 0.2

    def step(self):
        # Redistribute income
        if self.model.households:  # Check if there are households
            per_household = self.revenue / len(self.model.households)
            for h in self.model.households:
                h.wealth += per_household
        self.spending = self.revenue
        self.revenue = 0


class EconomyModel(Model):
    def __init__(self, N):
        super().__init__()
        self.num_households = N

        # Create agents
        self.households = []
        for _ in range(N):
            household = Household(self)
            self.households.append(household)

        self.firm = Firm(self)
        self.bank = Bank(self)
        self.government = Government(self)

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "Total Household Wealth": lambda m: sum([h.wealth for h in m.households]),
                "Government Revenue": lambda m: m.government.revenue,
                "Firm Profit": lambda m: m.firm.profit
            }
        )

        # Collect data for step 0
        self.datacollector.collect(self)

    def step(self):
        # Run everyone's step method
        for household in self.households:
            household.step()
        self.firm.step()
        self.bank.step()
        self.government.step()

        # Collect data
        self.datacollector.collect(self)


# Run the model
if __name__ == "__main__":
    model = EconomyModel(10)
    for i in range(50):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    print(df)