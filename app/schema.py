"""Input and output of the prediction API. Ranges follow the UCI dataset documentation."""
from pydantic import BaseModel, ConfigDict, Field

# Same order as ml/data.py FEATURES (a test checks they match).
FEATURES = [
    "limit_bal", "sex", "education", "marriage", "age",
    "pay_1", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
]

MAX_AMOUNT = 2_000_000  # NT dollars; the largest credit limit in the data is 1,000,000


def _pay_status(description: str):
    return Field(ge=-2, le=9, description=description)


def _bill():
    return Field(ge=-MAX_AMOUNT, le=MAX_AMOUNT, description="Bill statement amount (NT$)")


def _payment():
    return Field(ge=0, le=MAX_AMOUNT, description="Amount paid (NT$)")


class Client(BaseModel):
    """One credit card client. Unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid")

    limit_bal: float = Field(gt=0, le=MAX_AMOUNT, description="Credit limit (NT$)")
    sex: int = Field(ge=1, le=2, description="1 = male, 2 = female")
    education: int = Field(ge=0, le=6, description="1 graduate school, 2 university, 3 high school, 0/4-6 other")
    marriage: int = Field(ge=0, le=3, description="1 married, 2 single, 0/3 other")
    age: int = Field(ge=18, le=100)
    pay_1: int = _pay_status("Repayment status in September (-2..9, 1+ = months of delay)")
    pay_2: int = _pay_status("Repayment status in August")
    pay_3: int = _pay_status("Repayment status in July")
    pay_4: int = _pay_status("Repayment status in June")
    pay_5: int = _pay_status("Repayment status in May")
    pay_6: int = _pay_status("Repayment status in April")
    bill_amt1: float = _bill()
    bill_amt2: float = _bill()
    bill_amt3: float = _bill()
    bill_amt4: float = _bill()
    bill_amt5: float = _bill()
    bill_amt6: float = _bill()
    pay_amt1: float = _payment()
    pay_amt2: float = _payment()
    pay_amt3: float = _payment()
    pay_amt4: float = _payment()
    pay_amt5: float = _payment()
    pay_amt6: float = _payment()


class Prediction(BaseModel):
    probability_default: float
    prediction: int
    threshold: float
    model_name: str
    model_version: str


EXAMPLE_CLIENT = {
    "limit_bal": 50000, "sex": 2, "education": 2, "marriage": 1, "age": 35,
    "pay_1": 2, "pay_2": 2, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
    "bill_amt1": 46000, "bill_amt2": 45000, "bill_amt3": 44000,
    "bill_amt4": 30000, "bill_amt5": 29000, "bill_amt6": 28000,
    "pay_amt1": 0, "pay_amt2": 2000, "pay_amt3": 1500,
    "pay_amt4": 1000, "pay_amt5": 1000, "pay_amt6": 1000,
}
