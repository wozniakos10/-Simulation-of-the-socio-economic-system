import random

from RLAgent import RLAgent, POLISH_DATA_2024


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
