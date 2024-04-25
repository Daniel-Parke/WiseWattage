import logging
import numpy_financial as npf
import numpy as np
import pandas as pd

def initialise_finance(self):
    self.finance_model = calc_cashflow(self)

def calc_cashflow(self):
    total_payments = self.payments_per_year * self.loan_years
    total_principal = self.loan_amount + self.fees if self.upfront_fees else self.loan_amount
    
    # Calculate the monthly payment, total payment, and array of interest and principal payments
    rate_per_period = self.apr / self.payments_per_year
    monthly_payment = npf.pmt(rate_per_period, total_payments, -total_principal).round(2)
    total_payment = monthly_payment * total_payments
    
    interest_payments = npf.ipmt(rate_per_period, np.arange(1, total_payments + 1), total_payments, -total_principal).round(2)
    principal_payments = npf.ppmt(rate_per_period, np.arange(1, total_payments + 1), total_payments, -total_principal).round(2)

    # Calculate cumulative sums of payments
    cumulative_principal_payments = np.cumsum(principal_payments).round(2)
    cumulative_interest_payments = np.cumsum(interest_payments).round(2)
    
    # Remaining balances calculation
    remaining_balances = total_payment - cumulative_principal_payments - cumulative_interest_payments
    remaining_principals = total_principal + cumulative_principal_payments  # Reverse cumulative sum for principal
    remaining_interests = total_payment - cumulative_interest_payments  # Initial total payment minus cumulated interest payments
    
    # Prepare data for DataFrame
    data = np.column_stack((
        np.arange(total_payments + 1),  # Months from 0 to total_payments
        np.insert(monthly_payment * np.ones(total_payments), 0, 0),  # Monthly payments
        np.insert(interest_payments, 0, 0),  # Interest payments
        np.insert(principal_payments, 0, 0),  # Principal payments
        np.insert(remaining_balances, 0, total_payment),  # Remaining balance
        np.insert(remaining_principals, 0, total_principal),  # Remaining principal
        np.insert(remaining_interests, 0, total_payment)  # Remaining interest
    ))

    # Create DataFrame
    columns = ['Interval', 'Monthly Payment', 'Interest Payment', 'Principal Payment', 'Remaining Balance', 'Remaining Principal', 'Remaining Interest']
    df = pd.DataFrame(data, columns=columns)
    df["Interval"] = df["Interval"].round(0).astype(int)
    df = df.set_index("Interval").round(2)
    
    # Return the DataFrame
    return df
