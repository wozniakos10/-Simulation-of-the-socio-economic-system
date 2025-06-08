import random
import numpy as np

from RLAgent import RLAgent, POLISH_DATA_2024

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