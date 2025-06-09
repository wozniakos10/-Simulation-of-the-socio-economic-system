import random
import numpy as np

from RLAgent import RLAgent

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