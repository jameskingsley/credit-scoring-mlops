from pydantic import BaseModel

class CreditInput(BaseModel):
    Age: int
    EmploymentType: str
    ApplicantType: str
    AnnualIncome: float
    MonthlyIncome: float
    LoanType: str
    LoanAmount: float
    LoanTenureMonths: int
    InterestRate: float
    CollateralValue: float
    CreditScore: float
    PastDefaults: int
    NumOpenAccounts: int
    BusinessRevenue: float = 0
    ProfitMargin: float = 0
    BusinessYears: float = 0
