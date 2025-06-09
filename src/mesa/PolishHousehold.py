import random

from RLAgent import RLAgent, POLISH_DATA_2024


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
