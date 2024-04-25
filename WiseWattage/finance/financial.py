import logging
import numpy_financial as npf
import numpy as np
import pandas as pd

def initialise_finance(self):
    self.finance_model = calc_cashflow(self)

def calc_cashflow(self):
    total_payments = self.payments_per_year * self.loan_years
    total_principal = self.loan_amount + self.fees if self.upfront_fees else self.loan_amount

    # Create DataFrame with the interval index
    df = pd.DataFrame(index=np.arange(total_payments + 1))
    
    # Calculate the monthly payment
    df['Monthly Payment'] = npf.pmt(self.apr / self.payments_per_year, total_payments, -total_principal)
    df.loc[0, 'Monthly Payment'] = 0  # First month payment is 0
    
    # Calculate payments and remaining balances
    df['Principal Payment'] = npf.ppmt(self.apr / self.payments_per_year, df.index, total_payments, -total_principal)
    df['Interest Payment'] = npf.ipmt(self.apr / self.payments_per_year, df.index, total_payments, -total_principal)
    df.loc[0, ['Interest Payment', 'Principal Payment']] = 0  # First month payments are 0
    
    # Cumulative sums for principal and interest
    df['Cumulative Principal'] = df['Principal Payment'].cumsum()
    df['Cumulative Interest'] = df['Interest Payment'].cumsum()
    
    # Remaining balances
    df['Remaining Principal'] = total_principal - df['Cumulative Principal']
    df['Remaining Interest'] = np.sum(df['Interest Payment']) - df['Cumulative Interest']
    df['Remaining Balance'] = df['Remaining Principal'] + df['Remaining Interest']
    
    # Drop Cumulative columns and round values
    df.drop(['Cumulative Principal', 'Cumulative Interest'], axis=1, inplace=True)
    df = abs(df.round(2))  # Rounding to two decimal places
    
    return df