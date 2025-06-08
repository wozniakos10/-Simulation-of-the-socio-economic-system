import re

from mesa import Agent, Model
from mesa.datacollection import DataCollector
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Polskie dane ekonomiczne GUS 2024
POLISH_DATA_2024 = {
    'average_gross_wage': 8279,  # średnie wynagrodzenie brutto w zł (GUS lipiec 2024)
    'average_net_wage': 7140,    # średnie wynagrodzenie netto w zł (GUS 2024)
    'inflation_rate': 0.047,     # inflacja 4.7% (grudzień 2024)
    'pit_rate_1': 0.12,          # pierwszy próg podatkowy PIT
    'pit_rate_2': 0.32,          # drugi próg podatkowy PIT
    'pit_threshold': 120000,     # próg podatkowy w zł
    'tax_free_amount': 30000,    # kwota wolna od podatku
    'cit_rate': 0.19,            # CIT dla małych firm
    'vat_rate': 0.23,            # podstawowa stawka VAT
    'min_wage': 4300,            # minimalne wynagrodzenie 2024
    'unemployment_rate': 0.029,  # bezrobocie w Polsce
    'gdp_growth': 0.028          # wzrost PKB szacunkowy
}

class RLAgent(Agent, ABC):
    """Base class for RL agents"""
    def __init__(self, model):
        super().__init__(model)
        self.state = None
        self.action = None
        self.reward = 0
        self.prev_state = None
        self.prev_action = None
        self.reward_history = []

    @abstractmethod
    def get_state(self):
        """Returns the current state"""
        pass

    @abstractmethod
    def choose_action(self, state):
        """Chooses an action based on the state"""
        pass

    @abstractmethod
    def update_policy(self, state, action, reward, next_state):
        """Updates the learning policy"""
        pass

    def calculate_reward(self):
        """Calculates the reward for the current step"""
        return 0


class PolishCentralBank(RLAgent):
    """Central bank of Poland with reinforcement learning and real economic parameters."""

    def __init__(self, model):
        super().__init__(model)
        self.total_deposits = 0
        self.total_loans = 0
        self.profit = 0
        self.prev_profit = 0
        self.bad_loans = 0

        self.base_rate = 0.055  # NBP reference rate
        self.deposit_rate = 0.035
        self.loan_rate = 0.095
        self.reserve_ratio = 0.08
        self.reserves = 100000

        # Q-learning parameters
        self.learning_rate = 0.1
        self.epsilon = 0.15
        self.discount_factor = 0.9
        self.q_table = {}

        self.actions = ['conservative', 'moderate', 'aggressive', 'promotional']
        self.action_history = []

        self.min_credit_score = 650
        self.max_debt_ratio = 0.4

    def get_state(self):
        deposit_level = 'low' if self.total_deposits < 50000 else 'high'

        # Count loan applicants
        loan_applicants = sum(1 for h in self.model.households if getattr(h, 'wants_loan', False))
        loan_applicants += sum(1 for f in self.model.firms if getattr(f, 'wants_loan', False))

        loan_demand = 'low' if loan_applicants < 2 else 'high'
        inflation = 'high' if POLISH_DATA_2024['inflation_rate'] > 0.04 else 'low'


        return (deposit_level, loan_demand, inflation)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def apply_banking_policy(self, action):
        if action == 'conservative':
            self.min_credit_score = 700
            self.max_debt_ratio = 0.3
            self.loan_rate += 0.01
        elif action == 'moderate':
            self.min_credit_score = 650
            self.max_debt_ratio = 0.4
        elif action == 'aggressive':
            self.min_credit_score = 600
            self.max_debt_ratio = 0.5
            self.loan_rate -= 0.005
        elif action == 'promotional':
            self.deposit_rate += 0.01
            self.loan_rate -= 0.01

    def process_deposits(self):
        interest_paid = 0
        for household in self.model.households:
            if hasattr(household, 'bank_deposits') and household.bank_deposits > 0:
                interest = household.bank_deposits * (self.deposit_rate / 12)
                household.bank_deposits += interest
                household.wealth += interest
                interest_paid += interest
        self.total_deposits = sum(getattr(h, 'bank_deposits', 0) for h in self.model.households)
        return interest_paid

    def process_loans(self):
        """Process loan applications and existing loan payments"""
        loans_granted = 0
        interest_earned = 0

        # Step 1: Process NEW loan applications
        loan_applicants = []

        # Collect household applicants
        for household in self.model.households:
            if getattr(household, 'wants_loan', False):
                loan_applicants.append(household)

        # Collect firm applicants
        for firm in self.model.firms:
            if getattr(firm, 'wants_loan', False):
                loan_applicants.append(firm)

        # Process each loan application
        for applicant in loan_applicants:
            try:
                credit_score = getattr(applicant, 'credit_score', 650)

                # Determine income and loan amount
                if hasattr(applicant, 'net_wage'):  # Household
                    monthly_income = getattr(applicant, 'net_wage', 3000)
                    max_loan_amount = monthly_income * 24  # 2 years of income
                    loan_type = "personal"
                else:  # Firm
                    monthly_revenue = getattr(applicant, 'revenue', 10000) / 12
                    max_loan_amount = monthly_revenue * 12  # 1 year of revenue
                    loan_type = "business"

                # Limit loan amount
                loan_amount = min(max_loan_amount, 50000)  # Cap at 50k

                # Calculate current debt ratio
                current_debt = getattr(applicant, 'debt', 0)
                annual_income = monthly_income * 12 if hasattr(applicant, 'net_wage') else getattr(applicant, 'revenue', 120000)
                debt_ratio = current_debt / annual_income if annual_income > 0 else 1.0

                # Loan approval criteria
                if credit_score >= self.min_credit_score:

                    # Grant the loan
                    if hasattr(applicant, 'wealth'):  # Household
                        applicant.wealth += loan_amount
                    else:  # Firm - add to revenue for next period
                        if not hasattr(applicant, 'cash_injection'):
                            applicant.cash_injection = 0
                        applicant.cash_injection += loan_amount

                    # Record the debt
                    if not hasattr(applicant, 'debt'):
                        applicant.debt = 0
                    if not hasattr(applicant, 'monthly_payment'):
                        applicant.monthly_payment = 0

                    applicant.debt += loan_amount
                    # Calculate monthly payment (5% annual rate, 3 year term)
                    monthly_rate = 0.05 / 12
                    num_payments = 36
                    if monthly_rate > 0:
                        payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
                    else:
                        payment = loan_amount / num_payments
                    applicant.monthly_payment = getattr(applicant, 'monthly_payment', 0) + payment

                    # Update bank records
                    self.total_loans += loan_amount
                    self.reserves -= loan_amount
                    loans_granted += 1

                    # Mark as processed
                    applicant.wants_loan = False

                else:
                    applicant.wants_loan = False  # Don't keep trying every step

            except Exception as e:
                applicant.wants_loan = False

        # Step 2: Process EXISTING loan payments
        all_agents = list(self.model.households) + list(self.model.firms)

        for agent in all_agents:
            if hasattr(agent, 'debt') and agent.debt > 0:
                monthly_payment = getattr(agent, 'monthly_payment', 0)

                if monthly_payment > 0:
                    # Check if agent can make payment
                    available_funds = getattr(agent, 'wealth', 0)
                    if hasattr(agent, 'cash_injection'):
                        available_funds += agent.cash_injection

                    if available_funds >= monthly_payment:
                        # Make successful payment
                        if hasattr(agent, 'wealth'):
                            agent.wealth -= monthly_payment
                        else:
                            agent.cash_injection = getattr(agent, 'cash_injection', 0) - monthly_payment

                        # Split into interest and principal
                        interest_part = agent.debt * (0.05 / 12)  # 5% annual rate
                        principal_part = monthly_payment - interest_part

                        agent.debt -= principal_part
                        interest_earned += interest_part
                        self.reserves += monthly_payment

                        # Loan fully paid off
                        if agent.debt <= 0:
                            agent.debt = 0
                            agent.monthly_payment = 0
                            self.total_loans -= principal_part  # Remove from total loans

                    else:
                        self.bad_loans += monthly_payment
                        # Could add penalty or collection logic here

        return interest_earned, loans_granted

    def calculate_reward(self):
        profit_change = self.profit - self.prev_profit
        profit_reward = profit_change * 0.1
        deposit_growth = 10 if self.total_deposits > 200000 else 0
        bad_debt_penalty = -self.bad_loans * 0.5
        stability_bonus = 5 if self.reserves > 50000 else -10
        return profit_reward + deposit_growth + bad_debt_penalty + stability_bonus

    def update_policy(self, state, action, reward, next_state):
        self.q_table.setdefault(state, {a: 0.0 for a in self.actions})
        self.q_table.setdefault(next_state, {a: 0.0 for a in self.actions})
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def step(self):
        self.prev_state = getattr(self, 'state', None)
        self.prev_profit = self.profit

        self.state = self.get_state()
        self.action = self.choose_action(self.state)
        self.action_history.append(self.action)
        self.apply_banking_policy(self.action)

        # Process all banking operations
        interest_paid = self.process_deposits()
        interest_earned, loans_granted = self.process_loans()

        # Calculate profit
        operating_costs = 1000  # Basic operating costs
        self.profit = interest_earned - interest_paid - self.bad_loans * 0.1 - operating_costs
        self.reserves += self.profit

        # Ensure reserves don't go negative
        if self.reserves < 0:
            self.reserves = max(0, self.reserves)

        self.reward = self.calculate_reward()
        self.reward_history.append(self.reward)

        if self.prev_state is not None:
            self.update_policy(self.prev_state, getattr(self, 'prev_action', self.action), self.reward, self.state)

        self.prev_action = self.action

        # Reset bad_loans accumulator (but don't zero it completely)
        self.bad_loans *= 0.9  # Decay bad loans over time instead of zeroing


class PolishHousehold(RLAgent):
    """Polish household with realistic parameters"""
    def __init__(self, model, agent_id):
        super().__init__(model)
        self.agent_id = agent_id

        # Start with realistic values for Poland
        self.wealth = random.randint(10000, 50000)  # savings in PLN
        self.gross_wage = random.randint(4300, 15000)  # gross salary
        self.net_wage = self.calculate_net_wage(self.gross_wage)
        self.prev_wealth = self.wealth

        # Living costs in Poland
        self.monthly_costs = random.randint(2500, 4000)  # monthly living costs
        self.savings_rate = random.uniform(0.05, 0.15)   # savings rate

        # Employer assignment (will be set during model initialization)
        self.employer = None

        # RL parameters
        self.learning_rate = 0.1
        self.epsilon = 0.1
        self.discount_factor = 0.95

        # Q-table and action tracking
        self.q_table = {}
        self.actions = ['frugal', 'moderate', 'spendthrift']
        self.action_history = []
        self.current_savings = 0

    def calculate_net_wage(self, gross_wage):
        """Calculate net wage according to Polish tax rates"""
        # Simplified tax system (without ZUS contributions for clarity)
        annual_gross = gross_wage * 12

        if annual_gross <= POLISH_DATA_2024['tax_free_amount']:
            tax = 0
        elif annual_gross <= POLISH_DATA_2024['pit_threshold']:
            taxable = annual_gross - POLISH_DATA_2024['tax_free_amount']
            tax = taxable * POLISH_DATA_2024['pit_rate_1']
        else:
            tax_first = (POLISH_DATA_2024['pit_threshold'] - POLISH_DATA_2024['tax_free_amount']) * POLISH_DATA_2024['pit_rate_1']
            tax_second = (annual_gross - POLISH_DATA_2024['pit_threshold']) * POLISH_DATA_2024['pit_rate_2']
            tax = tax_first + tax_second

        monthly_tax = tax / 12
        return gross_wage - monthly_tax - (gross_wage * 0.1976)  # ZUS employee contributions ~19.76%

    def get_state(self):
        wealth_level = 'low' if self.wealth < 20000 else 'medium' if self.wealth < 50000 else 'high'
        wage_level = 'low' if self.net_wage < 4000 else 'medium' if self.net_wage < 8000 else 'high'
        inflation_impact = 'high' if POLISH_DATA_2024['inflation_rate'] > 0.04 else 'low'
        return (wealth_level, wage_level, inflation_impact)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def get_consumption_amount(self, action):
        disposable_income = max(0, self.net_wage - self.monthly_costs)

        if action == 'frugal':
            # High savings, low additional spending
            consumption = disposable_income * 0.3
            savings = disposable_income * 0.7
        elif action == 'moderate':
            # Balanced approach
            consumption = disposable_income * 0.6
            savings = disposable_income * 0.4
        else:  # 'spendthrift'
            # High spending, low savings
            consumption = disposable_income * 0.9
            savings = disposable_income * 0.1

        self.current_savings = max(0, savings)
        return max(0, consumption)

    def calculate_reward(self):
        wealth_change = self.wealth - self.prev_wealth

        # Reward for wealth growth adjusted for inflation
        real_wealth_change = wealth_change - (self.prev_wealth * POLISH_DATA_2024['inflation_rate'] / 12)

        # Bonus for financial stability
        stability_bonus = 10 if self.wealth > self.monthly_costs * 6 else 0  # 6-month cushion

        # Penalty for debt
        debt_penalty = -20 if self.wealth < 0 else 0

        return real_wealth_change + stability_bonus + debt_penalty

    def update_policy(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def step(self):
        self.prev_state = self.state
        self.prev_wealth = self.wealth

        self.state = self.get_state()
        self.action = self.choose_action(self.state)
        self.action_history.append(self.action)

        # Receive salary
        self.wealth += self.net_wage

        # Living costs (always)
        self.wealth -= self.monthly_costs

        # BANK INTERACTIONS - DEPOSITS
        if self.wealth > self.monthly_costs * 4:  # If comfortable savings
            deposit_amount = min(self.wealth * 0.05, 2000)  # More conservative deposits
            if not hasattr(self, 'bank_deposits'):
                self.bank_deposits = 0
            self.bank_deposits += deposit_amount
            self.wealth -= deposit_amount

        # BANK INTERACTIONS - LOAN REQUESTS (More likely to request loans)
        if not getattr(self, 'wants_loan', False) and not hasattr(self, 'debt'):
            # Request loan in various scenarios
            if (self.wealth < self.monthly_costs * 1.5 or  # Low funds
                    self.action == 'big_purchase' or  # Want to buy something
                    (self.wealth < 10000 and random.random() < 0.3)):  # Random need

                self.wants_loan = True
                if not hasattr(self, 'credit_score'):
                    self.credit_score = random.randint(580, 780)

        # Additional spending/savings
        consumption = self.get_consumption_amount(self.action)
        self.wealth -= consumption
        self.wealth += self.current_savings

        # Transfer consumption to the household's specific employer
        if self.employer is not None:
            self.employer.consumer_spending += consumption
        else:
            consumption_per_firm = consumption / len(self.model.firms) if self.model.firms else 0
            for firm in self.model.firms:
                firm.consumer_spending += consumption_per_firm

        # Inflation effect
        inflation_loss = self.wealth * (POLISH_DATA_2024['inflation_rate'] / 12)
        self.wealth -= inflation_loss

        # RL update
        self.reward = self.calculate_reward()
        self.reward_history.append(self.reward)

        if self.prev_state is not None:
            self.update_policy(self.prev_state, self.prev_action, self.reward, self.state)

        self.prev_action = self.action



class PolishFirm(RLAgent):
    """Polish firm with realistic parameters"""
    def __init__(self, model, firm_id):
        super().__init__(model)
        self.firm_id = firm_id
        self.revenue = 0
        self.profit = 0
        self.prev_profit = 0
        self.consumer_spending = 0
        self.employees = []  # only assigned employees

        # Polish business parameters
        self.base_costs = random.randint(50000, 100000)
        self.tax_rate = POLISH_DATA_2024['cit_rate']

        # RL parameters
        self.learning_rate = 0.1
        self.epsilon = 0.15
        self.discount_factor = 0.9

        self.q_table = {}
        self.actions = ['low_wage', 'medium_wage', 'high_wage', 'bonuses']
        self.action_history = []


    def get_state(self):
        revenue_level = 'low' if self.revenue < 100000 else 'medium' if self.revenue < 300000 else 'high'
        demand_level = 'low' if self.consumer_spending < 50000 else 'high'
        market_condition = 'hard' if POLISH_DATA_2024['inflation_rate'] > 0.04 else 'stable'
        return (revenue_level, demand_level, market_condition)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def apply_wage_policy(self, action):
        for h in self.employees:
            if action == 'low_wage':
                h.gross_wage = max(POLISH_DATA_2024['min_wage'], h.gross_wage * 0.95)
            elif action == 'medium_wage':
                h.gross_wage = min(h.gross_wage * 1.02, POLISH_DATA_2024['average_gross_wage'])
            elif action == 'high_wage':
                h.gross_wage *= 1.05
            elif action == 'bonuses':
                h.wealth += random.randint(500, 2000)
            h.net_wage = h.calculate_net_wage(h.gross_wage)

    def calculate_reward(self):
        profit_change = self.profit - self.prev_profit
        profit_reward = profit_change * 0.1

        avg_wealth = np.mean([h.wealth for h in self.employees]) if self.employees else 0
        employee_satisfaction = 5 if avg_wealth > 30000 else 0

        total_wages = sum(h.gross_wage for h in self.employees)
        wage_efficiency = -0.01 * total_wages if total_wages > 200000 else 0

        return profit_reward + employee_satisfaction + wage_efficiency

    def update_policy(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def step(self):
        self.prev_state = self.state
        self.prev_profit = self.profit

        # Add any cash injection from loans to revenue
        if hasattr(self, 'cash_injection') and self.cash_injection > 0:
            self.revenue += self.cash_injection
            self.cash_injection = 0  # Use it up

        self.revenue += self.consumer_spending  # Regular consumer spending
        self.consumer_spending = 0

        self.state = self.get_state()
        self.action = self.choose_action(self.state)
        self.action_history.append(self.action)

        self.apply_wage_policy(self.action)

        wage_bill = sum(h.gross_wage * 1.2 for h in self.employees)
        total_costs = self.base_costs + wage_bill
        gross_profit = max(0, self.revenue - total_costs)

        # BANK INTERACTIONS - REQUEST LOANS when struggling
        if not getattr(self, 'wants_loan', False) and not hasattr(self, 'debt'):
            if (gross_profit < total_costs * 0.3 or  # Poor performance
                    self.revenue < total_costs or  # Losing money
                    random.random() < 0.2):  # Random business need

                self.wants_loan = True
                if not hasattr(self, 'credit_score'):
                    self.credit_score = random.randint(620, 750)

        # BANK INTERACTIONS - DEPOSITS (only if very profitable)
        if gross_profit > total_costs * 2:  # Very profitable
            deposit_amount = gross_profit * 0.1
            if not hasattr(self, 'bank_deposits'):
                self.bank_deposits = 0
            self.bank_deposits += deposit_amount
            gross_profit -= deposit_amount

        tax = gross_profit * self.tax_rate
        self.profit = gross_profit - tax
        self.model.government.tax_revenue += tax

        self.reward = self.calculate_reward()
        self.reward_history.append(self.reward)

        if self.prev_state is not None:
            self.update_policy(self.prev_state, self.prev_action, self.reward, self.state)

        self.prev_action = self.action



class PolishGovernment(RLAgent):
    """Polish government with realistic fiscal parameters"""
    def __init__(self, model):
        super().__init__(model)
        self.tax_revenue = 0
        self.spending = 0
        self.budget_balance = 0
        self.debt = 0

        # Polish fiscal parameters
        self.gdp_estimate = 3000000000  # estimated GDP in model scale
        self.debt_to_gdp_limit = 0.6    # constitutional debt limit

        # ENHANCED RL PARAMETERS
        self.learning_rate = 0.3  # increased learning speed
        self.epsilon = 0.4  # increased exploration (decreases over time)
        self.epsilon_decay = 0.995  # gradual exploration decrease
        self.epsilon_min = 0.1
        self.discount_factor = 0.85  # reduced for faster learning

        # Q-table initialization with optimistic values
        self.q_table = {}
        self.actions = ['austerity', 'balanced', 'expansive', 'investment']
        self.action_history = []

        # Additional parameters for performance tracking
        self.steps_taken = 0
        self.last_rewards = []
        self.action_counts = {action: 0 for action in self.actions}

        # Starting bonus for action diversity
        self.diversity_bonus = True

    def get_state(self):
        # Extended state representation
        budget_state = 'deficit' if self.budget_balance < -50000 else 'surplus' if self.budget_balance > 50000 else 'balanced'
        debt_level = 'high' if (self.debt / self.gdp_estimate) > 0.4 else 'medium' if (self.debt / self.gdp_estimate) > 0.2 else 'low'

        # Include household state
        avg_household_wealth = np.mean([h.wealth for h in self.model.households])
        social_state = 'poor' if avg_household_wealth < 25000 else 'middle' if avg_household_wealth < 40000 else 'wealthy'

        return (budget_state, debt_level, social_state)

    def choose_action(self, state):
        if state not in self.q_table:
            # Initialize with optimistic values instead of zeros
            self.q_table[state] = {action: random.uniform(5, 15) for action in self.actions}

        self.steps_taken += 1

        # Epsilon-greedy with diversity bonus
        if random.random() < self.epsilon:
            # During exploration, prefer less frequently used actions
            if self.diversity_bonus and self.steps_taken < 100:
                action_probs = []
                for action in self.actions:
                    # Less frequently used actions have higher chance
                    prob = 1.0 / (self.action_counts[action] + 1)
                    action_probs.append(prob)

                # Normalize probabilities
                total_prob = sum(action_probs)
                action_probs = [p/total_prob for p in action_probs]

                chosen_action = np.random.choice(self.actions, p=action_probs)
            else:
                chosen_action = random.choice(self.actions)
        else:
            # Choose best action with small noise
            q_values = self.q_table[state]
            best_actions = [action for action, value in q_values.items()
                            if value == max(q_values.values())]
            chosen_action = random.choice(best_actions)

        self.action_counts[chosen_action] += 1

        # Gradually decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return chosen_action

    def apply_fiscal_policy(self, action):
        base_spending = max(self.tax_revenue * 0.8, 50000)  # minimum spending

        if action == 'austerity':
            self.spending = base_spending * 0.7
            # Cuts in transfers
            for h in self.model.households:
                if h.wealth < 15000:  # help only for the poorest
                    h.wealth += 200

        elif action == 'balanced':
            self.spending = base_spending * 1.0
            # Standard social transfers
            for h in self.model.households:
                if h.wealth < 25000:
                    h.wealth += 500

        elif action == 'expansive':
            self.spending = base_spending * 1.4
            # Increased transfers and social programs
            for h in self.model.households:
                h.wealth += 800  # 500+ type program
                if h.wealth < 20000:  # additional help for poor
                    h.wealth += 600

        elif action == 'investment':
            self.spending = base_spending * 1.3
            # Public investments stimulating economy
            investment_boost = base_spending * 0.3

            # Distribute public investment among all firms
            investment_per_firm = investment_boost / len(self.model.firms)
            for firm in self.model.firms:
                firm.revenue += investment_per_firm  # public orders

            # Investment effect - job creation
            for h in self.model.households:
                if random.random() < 0.4:  # 40% chance of benefiting from investment
                    h.wealth += random.randint(1500, 4000)
                    h.gross_wage *= 1.02  # wage growth through investments
                    h.net_wage = h.calculate_net_wage(h.gross_wage)

    def calculate_reward(self):
        # More balanced reward function

        # 1. Social welfare (50% weight)
        total_household_wealth = sum([h.wealth for h in self.model.households])
        avg_household_wealth = total_household_wealth / len(self.model.households)

        # Reward for higher average wealth, but with diminishing returns
        wealth_reward = np.log(max(avg_household_wealth, 1000)) * 20

        # Bonus for reducing inequality
        wealth_std = np.std([h.wealth for h in self.model.households])
        inequality_penalty = -wealth_std * 0.01

        # 2. Budget balance (25% weight)
        # Slight preference for balanced budget, but not too high penalty
        budget_reward = 0
        if abs(self.budget_balance) < 10000:  # balanced budget
            budget_reward = 20
        elif self.budget_balance > 50000:  # large surplus
            budget_reward = 10  # smaller reward - money could have been better used
        else:  # deficit
            budget_reward = -abs(self.budget_balance) * 0.001  # mild penalty

        # 3. Debt control (15% weight)
        debt_ratio = self.debt / self.gdp_estimate
        if debt_ratio > self.debt_to_gdp_limit:
            debt_penalty = -200  # high penalty for exceeding limit
        elif debt_ratio > 0.5:
            debt_penalty = -50   # warning
        else:
            debt_penalty = 0

        # 4. Economic growth (10% weight) - now considers all firms
        total_firm_profit = sum([firm.profit for firm in self.model.firms])
        economic_growth = max(total_firm_profit * 0.005, 0)

        # 5. Action diversity bonus (at the beginning)
        diversity_bonus = 0
        if self.steps_taken < 50:
            unique_actions = len(set(self.action_history[-10:]) if len(self.action_history) >= 10 else set(self.action_history))
            diversity_bonus = unique_actions * 3

        total_reward = wealth_reward + inequality_penalty + budget_reward + debt_penalty + economic_growth + diversity_bonus

        # Save reward to history
        self.last_rewards.append(total_reward)
        if len(self.last_rewards) > 20:
            self.last_rewards.pop(0)

        return total_reward

    def update_policy(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: random.uniform(5, 15) for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: random.uniform(5, 15) for a in self.actions}

        # Double Q-Learning for better stability
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]

        # Adaptive learning rate
        adaptive_lr = self.learning_rate * (1.0 / (1.0 + self.steps_taken * 0.001))
        self.q_table[state][action] += adaptive_lr * td_error

        # Clip Q-values for stability
        for action_key in self.q_table[state]:
            self.q_table[state][action_key] = np.clip(self.q_table[state][action_key], -100, 100)

    def step(self):
        self.prev_state = self.state

        self.state = self.get_state()
        self.action = self.choose_action(self.state)
        self.action_history.append(self.action)

        self.apply_fiscal_policy(self.action)

        # Budget balance
        self.budget_balance = self.tax_revenue - self.spending

        # Debt update
        if self.budget_balance < 0:
            self.debt += abs(self.budget_balance)
        else:
            self.debt = max(0, self.debt - self.budget_balance * 0.5)  # debt repayment

        # RL update
        self.reward = self.calculate_reward()
        self.reward_history.append(self.reward)

        if self.prev_state is not None and self.prev_action is not None:
            self.update_policy(self.prev_state, self.prev_action, self.reward, self.state)

        self.prev_action = self.action
        self.tax_revenue = 0  # reset for next period

    def get_policy_summary(self):
        """Returns summary of learned policy"""
        print("\n=== GOVERNMENT POLICY SUMMARY ===")
        print(f"Steps taken: {self.steps_taken}")
        print(f"Current epsilon: {self.epsilon:.3f}")
        print(f"Average reward (last 20 steps): {np.mean(self.last_rewards) if self.last_rewards else 0:.2f}")

        print("\nAction selection frequency:")
        total_actions = sum(self.action_counts.values())
        for action, count in self.action_counts.items():
            percentage = (count/total_actions)*100 if total_actions > 0 else 0
            print(f"  {action}: {count} times ({percentage:.1f}%)")

        print("\nLearned Q-values (for main states):")
        for state, q_values in list(self.q_table.items())[:5]:  # show only first 5 states
            print(f"  State {state}:")
            for action, value in q_values.items():
                print(f"    {action}: {value:.2f}")


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


# Run the enhanced simulation
if __name__ == "__main__":
    print("Starting Enhanced RL Economy Simulation...")
    print("This may take a moment to complete...")
    
    # Initialize model
    model = PolishEconomyModel(1000,50)  # Slightly fewer households for cleaner visualization
    
    # Run simulation
    steps = 60  # More steps for better learning
    for i in range(steps):
        model.step()
        if (i + 1) % 30 == 0:
            print(f"Step {i + 1}/{steps} completed")
    
    print("Simulation completed!")
    print("\n" + "="*50)
    
    
    print("\nGenerating comprehensive visualizations...")
    model.visualize_polish_results()
    
    print("Analysis complete! Check the generated plots above.")