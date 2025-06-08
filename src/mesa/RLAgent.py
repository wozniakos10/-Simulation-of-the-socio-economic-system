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
