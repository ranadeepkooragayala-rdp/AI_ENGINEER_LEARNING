from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError

class LineItem(BaseModel):
    item_name: str = Field(min_length=1, max_length=100)
    quantity: int = Field(gt=0)
    unit_price: float = Field(gt=0.0)

    @property
    def total(self) -> float:
        return round(self.quantity * self.unit_price, 2)


class Customer(BaseModel):
    name: str = Field(min_length=2)
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format: must contain '@' and a domain with '.'")
        return v


class Invoice(BaseModel):
    invoice_id: str = Field(min_length=5)
    customer: Customer
    status: Literal["draft", "paid", "overdue"] = "draft"
    items: List[LineItem] = Field(min_length=1)
    discount_percentage: float = Field(default=0.0, ge=0.0, le=100.0)

    @field_validator("invoice_id")
    @classmethod
    def validate_invoice_id(cls, v: str) -> str:
        if not v.startswith("INV-"):
            raise ValueError("Invalid Invoice ID: must start with 'INV-'")
        return v

    def grand_total(self) -> float:
        sub_total = sum(item.total for item in self.items)
        discount = sub_total * (self.discount_percentage / 100.0)
        return round(sub_total - discount, 2)
