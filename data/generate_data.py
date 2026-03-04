"""
Synthetic business data generator.

Generates three tables of realistic business data and saves them to
``data/business_data.db`` (SQLite).  Run this script once before starting
the dashboard or notebooks::

    python data/generate_data.py
"""

import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "business_data.db"
SEED = 42
random.seed(SEED)
fake = Faker()
fake.seed_instance(SEED)

REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"]
CATEGORIES = ["Electronics", "Software", "Services", "Hardware", "Consulting"]
SEGMENTS = ["Enterprise", "SMB", "Startup", "Government"]
ACTIVITIES = [
    "Order Received",
    "Credit Check",
    "Inventory Check",
    "Fulfillment",
    "Shipping",
    "Delivered",
]

REGION_COUNTRIES = {
    "North America": ["United States", "Canada", "Mexico"],
    "Europe": ["Germany", "United Kingdom", "France", "Netherlands", "Spain"],
    "Asia-Pacific": ["Japan", "China", "Australia", "India", "South Korea"],
    "Latin America": ["Brazil", "Argentina", "Colombia", "Chile"],
    "Middle East": ["UAE", "Saudi Arabia", "Israel", "Qatar"],
}

REVIEW_TEMPLATES = {
    "positive": [
        "Excellent product, exceeded all expectations!",
        "Very satisfied with the quality and fast delivery.",
        "Outstanding support team, will definitely buy again.",
        "The {category} solution is exactly what we needed.",
        "Highly recommend to any {segment} company looking for value.",
        "Smooth onboarding and great feature set.",
    ],
    "neutral": [
        "Product works as described, nothing special.",
        "Delivery was on time, product is average.",
        "Does the job but could use more features.",
        "Acceptable quality for the price point.",
        "Support response was acceptable, not great.",
    ],
    "negative": [
        "Very disappointed with the product quality.",
        "Support team was unresponsive and unhelpful.",
        "Product did not meet our requirements at all.",
        "Numerous bugs and the documentation is poor.",
        "Would not recommend — misleading product description.",
        "Poor value for money and slow delivery.",
    ],
}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_transactions(n: int = 1200) -> pd.DataFrame:
    """Generate synthetic transactions table with *n* rows."""
    rows = []
    start = datetime(2024, 1, 1)
    end = datetime(2025, 12, 31)
    for _ in range(n):
        region = random.choice(REGIONS)
        category = random.choice(CATEGORIES)
        segment = random.choice(SEGMENTS)
        revenue = round(random.uniform(100, 50000), 2)
        cost = round(random.uniform(50, min(40000, revenue * 0.9)), 2)
        rows.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "date": fake.date_time_between(start_date=start, end_date=end).strftime(
                    "%Y-%m-%d"
                ),
                "region": region,
                "product_category": category,
                "customer_segment": segment,
                "revenue": revenue,
                "cost": cost,
                "units_sold": random.randint(1, 100),
                "sales_rep": fake.name(),
                "country": random.choice(REGION_COUNTRIES[region]),
            }
        )
    return pd.DataFrame(rows)


def generate_process_events(n_cases: int = 500) -> pd.DataFrame:
    """Generate synthetic process events table for *n_cases* unique cases."""
    rows = []
    for _ in range(n_cases):
        case_id = str(uuid.uuid4())
        ts = fake.date_time_between(start_date=datetime(2024, 1, 1), end_date=datetime(2025, 6, 1))
        for activity in ACTIVITIES:
            # Realistic base durations (minutes) per activity
            base_durations = {
                "Order Received": (5, 30),
                "Credit Check": (30, 120),
                "Inventory Check": (20, 90),
                "Fulfillment": (60, 480),
                "Shipping": (120, 720),
                "Delivered": (5, 60),
            }
            lo, hi = base_durations[activity]
            duration = round(random.uniform(lo, hi), 2)
            # Introduce occasional bottleneck anomalies for Fulfillment / Shipping
            if activity in ("Fulfillment", "Shipping") and random.random() < 0.12:
                duration = round(random.uniform(1440, 4320), 2)  # 1-3 days

            rows.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "case_id": case_id,
                    "activity": activity,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "resource": fake.name(),
                    "duration_minutes": duration,
                }
            )
            ts += timedelta(minutes=duration)
    return pd.DataFrame(rows)


def generate_customer_reviews(n: int = 600) -> pd.DataFrame:
    """Generate synthetic customer reviews table with *n* rows."""
    rows = []
    start = datetime(2024, 1, 1)
    end = datetime(2025, 12, 31)
    for _ in range(n):
        rating = random.randint(1, 5)
        if rating >= 4:
            sentiment = "positive"
        elif rating == 3:
            sentiment = "neutral"
        else:
            sentiment = "negative"

        category = random.choice(CATEGORIES)
        segment = random.choice(SEGMENTS)
        template = random.choice(REVIEW_TEMPLATES[sentiment])
        review_text = template.format(category=category.lower(), segment=segment.lower())

        rows.append(
            {
                "review_id": str(uuid.uuid4()),
                "date": fake.date_time_between(start_date=start, end_date=end).strftime(
                    "%Y-%m-%d"
                ),
                "product_category": category,
                "customer_segment": segment,
                "region": random.choice(REGIONS),
                "review_text": review_text,
                "rating": rating,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all tables and persist them to SQLite."""
    # Ensure the data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Import here to avoid circular import when modules.db is imported first
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{DB_PATH}")

    print("Generating data …")
    transactions = generate_transactions(1200)
    process_events = generate_process_events(500)
    reviews = generate_customer_reviews(600)

    transactions.to_sql("transactions", engine, if_exists="replace", index=False)
    process_events.to_sql("process_events", engine, if_exists="replace", index=False)
    reviews.to_sql("customer_reviews", engine, if_exists="replace", index=False)

    print("\n✅ Data generation complete")
    print(f"   transactions    : {len(transactions):,} rows")
    print(f"   process_events  : {len(process_events):,} rows")
    print(f"   customer_reviews: {len(reviews):,} rows")
    print(f"\n   Saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
