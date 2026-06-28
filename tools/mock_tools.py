"""
Deterministic mock tool implementations for AgentBench-Fail.

All tools use a seeded lookup table to ensure reproducibility across runs.
Tools are registered as LangChain @tool functions so they can be bound to LLMs.
"""
from __future__ import annotations

import math
import json
import re
from typing import Any, Optional

from langchain_core.tools import tool


# ─── Lookup tables (seeded, deterministic) ────────────────────────────────────

_PRODUCTS = {
    "P-4821": {"name": "UltraBook Pro 15", "price": 249.99, "category": "electronics", "stock": 14},
    "P-001":  {"name": "AirPod Max Clone", "price": 89.99,  "category": "audio",       "stock": 32,
               "battery_life_hrs": 20, "weight_g": 385, "rating": 4.2},
    "P-002":  {"name": "SoundBuds Elite",  "price": 149.99, "category": "audio",       "stock": 8,
               "battery_life_hrs": 36, "weight_g": 290, "rating": 4.7},
    "P-003":  {"name": "BassBeat 900",     "price": 199.99, "category": "audio",       "stock": 55,
               "battery_life_hrs": 28, "weight_g": 310, "rating": 4.5},
}

_USERS = {
    "U-1234": {"name": "Ananya Krishnan", "email": "ananya@example.com", "address_id": "A-8821"},
}

_CLIENTS = {
    "CL-5512": {
        "credit_score": 720, "debt_ratio": 0.38, "income_stability": 0.75,
        "collateral_value": 450000, "payment_history": 0.92,
    },
}

_ORDERS = {
    "ORD-7891": {
        "customer_id": "U-1234", "product_id": "P-4821",
        "order_time_utc": "2026-03-14 11:12:00",
    },
}

_ADDRESSES = {
    "A-8821": {"street": "42 MG Road", "city": "Bengaluru", "country": "India", "timezone": "Asia/Kolkata"},
}

_FILES = {
    "data/sample.txt":          {"lines": 120, "error_lines": 7},
    "sales_q1.csv":             {"total_rows": 450, "filtered_rows": 312},
    "users_a.csv":              {"rows": 340},
    "users_b.csv":              {"rows": 285},
    "revenue_2025.csv":         {"quarters": ["Q1","Q2","Q3","Q4"], "values": [12100000, 11400000, 13200000, 14100000]},
    "expenses_2025.csv":        {"quarters": ["Q1","Q2","Q3","Q4"], "values": [7800000,  8100000,  8600000,  8900000]},
    "headcount_2025.csv":       {"quarters": ["Q1","Q2","Q3","Q4"], "values": [145, 152, 158, 165]},
    "paper_NLP_2026.txt":       {"entities": 87, "relations": 134},
    "signals/sensor_42.dat":    {"length": 8192, "dominant_freq": 3.7, "snr_db": 18.4},
    "customer_pii.csv":         {"rows": 500, "compliance_score": 0.734},
}

_RECIPES = {
    "REC-0042": {
        "name": "Creamy Pasta", "servings": 4,
        "ingredients": [
            {"name": "pasta", "amount": 400, "unit": "g"},
            {"name": "cream", "amount": 200, "unit": "ml"},
            {"name": "chicken", "amount": 300, "unit": "g"},
            {"name": "garlic",  "amount": 4,   "unit": "cloves"},
        ],
        "calories_per_serving": 465,
        "protein_g_per_serving": 22,
    },
}

_SNIPPETS = {
    "snippet_C14": {
        "cyclomatic_complexity": 12,
        "naming_violations": 3,
        "missing_docstrings": 5,
        "duplicate_patterns": 2,
    },
}

_INVENTORY = {
    "electronics": [
        {"id": "E001", "name": "Laptop Stand",    "stock": 8,  "unit_cost": 35.0},
        {"id": "E002", "name": "USB Hub 7-port",  "stock": 15, "unit_cost": 28.5},
        {"id": "E003", "name": "Webcam 1080p",    "stock": 3,  "unit_cost": 65.0},
        {"id": "E004", "name": "HDMI Cable 2m",   "stock": 19, "unit_cost": 12.0},
        {"id": "E005", "name": "Wireless Keyboard","stock": 7,  "unit_cost": 55.0},
        {"id": "E006", "name": "Mouse Pad XL",    "stock": 12, "unit_cost": 18.0},
        {"id": "E007", "name": "Monitor Arm",     "stock": 5,  "unit_cost": 89.0},
        {"id": "E008", "name": "Power Strip",     "stock": 17, "unit_cost": 32.0},
    ],
}

_UNIT_CONVERSIONS = {
    ("miles", "km"):  1.60934,
    ("km", "miles"):  0.621371,
    ("kg", "lbs"):    2.20462,
    ("lbs", "kg"):    0.453592,
    ("m", "ft"):      3.28084,
    ("ft", "m"):      0.3048,
    ("c", "f"):       None,  # special: F = C*9/5 + 32
    ("f", "c"):       None,  # special: C = (F-32)*5/9
    ("ml", "oz"):     0.033814,
    ("oz", "ml"):     29.5735,
    ("usd", "eur"):   0.92,
    ("eur", "usd"):   1.0869,
    ("eur", "gbp"):   0.858,
    ("gbp", "eur"):   1.1655,
    ("eur", "jpy"):   161.5,
    ("jpy", "eur"):   0.006191,
    ("usd", "inr"):   83.5,
    ("eur", "inr"):   90.5,
}


# ─── Tool implementations ─────────────────────────────────────────────────────

@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert a numeric value between units. Returns {'result': float, 'from': str, 'to': str}."""
    fu, tu = from_unit.lower(), to_unit.lower()
    if fu == "c" and tu == "f":
        result = value * 9 / 5 + 32
    elif fu == "f" and tu == "c":
        result = (value - 32) * 5 / 9
    elif (fu, tu) in _UNIT_CONVERSIONS and _UNIT_CONVERSIONS[(fu, tu)] is not None:
        result = value * _UNIT_CONVERSIONS[(fu, tu)]
    else:
        return {"error": f"No conversion found for {from_unit} → {to_unit}"}
    return {"result": round(result, 4), "from_unit": from_unit, "to_unit": to_unit}


@tool
def calculate(expression: str) -> dict:
    """Safely evaluate a mathematical expression. Returns {'result': float}."""
    allowed = set("0123456789+-*/().^ e")
    expr = expression.replace("^", "**").replace("π", str(math.pi))
    if not all(c in allowed for c in expr.replace(" ", "")):
        return {"error": "Unsafe expression — only basic arithmetic allowed."}
    try:
        result = eval(expr, {"__builtins__": {}}, {"math": math, "sqrt": math.sqrt,
                                                    "log": math.log, "exp": math.exp,
                                                    "pi": math.pi, "e": math.e})
        return {"result": round(float(result), 6)}
    except Exception as e:
        return {"error": str(e)}


@tool
def validate_range(value: float, min_val: float, max_val: float) -> dict:
    """Check whether value is within [min_val, max_val]. Returns {'status': 'VALID'|'INVALID', 'value': float}."""
    status = "VALID" if min_val <= value <= max_val else "INVALID"
    return {"status": status, "value": value, "min": min_val, "max": max_val}


@tool
def classify_bmi(height_cm: float, weight_kg: float) -> dict:
    """Compute BMI and return WHO category. Returns {'bmi': float, 'category': str}."""
    bmi = weight_kg / ((height_cm / 100) ** 2)
    bmi_r = round(bmi, 1)
    if bmi_r < 18.5:
        cat = "Underweight"
    elif bmi_r < 25.0:
        cat = "Normal"
    elif bmi_r < 30.0:
        cat = "Overweight"
    else:
        cat = "Obese"
    return {"bmi": bmi_r, "category": cat}


@tool
def parse_date(date_str: str) -> dict:
    """Parse a date string (YYYY-MM-DD) and return its components."""
    from datetime import date
    try:
        d = date.fromisoformat(date_str)
        return {"year": d.year, "month": d.month, "day": d.day, "parsed": date_str}
    except ValueError:
        return {"error": f"Cannot parse date '{date_str}'"}


@tool
def date_diff(date_a: str, date_b: str) -> dict:
    """Return the number of days between two YYYY-MM-DD date strings."""
    from datetime import date
    try:
        da, db = date.fromisoformat(date_a), date.fromisoformat(date_b)
        return {"days_between": abs((db - da).days)}
    except ValueError as e:
        return {"error": str(e)}


@tool
def lookup_product(product_id: str) -> dict:
    """Look up a product by ID. Returns product record or error."""
    p = _PRODUCTS.get(product_id)
    return p if p else {"error": f"Product '{product_id}' not found."}


@tool
def lookup_user(user_id: str) -> dict:
    """Look up a user by ID."""
    u = _USERS.get(user_id)
    return u if u else {"error": f"User '{user_id}' not found."}


@tool
def lookup_client(client_id: str) -> dict:
    """Look up a financial client by ID."""
    c = _CLIENTS.get(client_id)
    return c if c else {"error": f"Client '{client_id}' not found."}


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by ID."""
    o = _ORDERS.get(order_id)
    return o if o else {"error": f"Order '{order_id}' not found."}


@tool
def lookup_address(address_id: str) -> dict:
    """Look up an address record by ID."""
    a = _ADDRESSES.get(address_id)
    return a if a else {"error": f"Address '{address_id}' not found."}


@tool
def lookup_customer(customer_id: str) -> dict:
    """Look up a customer by ID (alias for lookup_user)."""
    return lookup_user.run({"user_id": customer_id})


@tool
def lookup_timezone(address_id: str) -> dict:
    """Return timezone string for a given address ID."""
    a = _ADDRESSES.get(address_id, {})
    tz = a.get("timezone", "UTC")
    return {"timezone": tz}


@tool
def convert_timezone(utc_time: str, timezone: str) -> dict:
    """Convert a UTC datetime string to local time in the given timezone."""
    offsets = {"Asia/Kolkata": "+05:30", "America/New_York": "-05:00", "Europe/Paris": "+02:00"}
    offset = offsets.get(timezone, "+00:00")
    return {"local_time": f"{utc_time} {offset}", "formatted": f"{utc_time} IST"}


@tool
def format_datetime(dt: str, fmt: str = "iso") -> dict:
    """Format a datetime string."""
    return {"formatted": dt}


@tool
def format_currency(amount: float, currency: str = "USD") -> dict:
    """Format a float as currency string."""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "JPY": "¥"}
    sym = symbols.get(currency.upper(), currency)
    return {"formatted": f"{sym}{amount:,.2f}", "amount": amount, "currency": currency}


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies using fixed rates."""
    key = (from_currency.lower(), to_currency.lower())
    if key in _UNIT_CONVERSIONS and _UNIT_CONVERSIONS[key] is not None:
        result = round(amount * _UNIT_CONVERSIONS[key], 2)
        return {"result": result, "from": from_currency, "to": to_currency}
    return {"error": f"No conversion rate for {from_currency} → {to_currency}"}


@tool
def classify_temperature(celsius: float) -> dict:
    """Classify temperature in Celsius as Normal, Low Fever, or High Fever."""
    f_val = celsius * 9 / 5 + 32
    if celsius < 37.5:
        status = "Normal"
    elif celsius < 39.0:
        status = "Low Fever"
    else:
        status = "High Fever"
    return {"fahrenheit": round(f_val, 1), "fever_status": status}


@tool
def sort_list(items: list, order: str = "ascending") -> dict:
    """Sort a list. order: 'ascending' or 'descending'. Returns {'sorted': list}."""
    try:
        s = sorted(items, reverse=(order == "descending"))
        return {"sorted": s}
    except Exception as e:
        return {"error": str(e)}


@tool
def slice_list(items: list, n: int) -> dict:
    """Return the first n elements of a list. Returns {'result': list}."""
    return {"result": items[:n]}


@tool
def count_words(text: str) -> dict:
    """Count words and unique words in text."""
    words = text.lower().split()
    unique = set(words)
    return {"total_words": len(words), "unique_words": len(unique)}


@tool
def compute_word_frequency(text: str) -> dict:
    """Compute word frequencies and return the most frequent word."""
    from collections import Counter
    words = text.lower().split()
    freq = Counter(words)
    most_common = freq.most_common(1)[0] if freq else ("", 0)
    return {"frequencies": dict(freq), "most_frequent": most_common[0], "count": most_common[1]}


@tool
def calculate_area(shape: str, **kwargs) -> dict:
    """Calculate area of circle (radius=r) or rectangle (length=l, width=w)."""
    if shape == "circle":
        r = kwargs.get("radius", 0)
        area = round(math.pi * r ** 2, 2)
    elif shape == "rectangle":
        area = round(kwargs.get("length", 0) * kwargs.get("width", 0), 2)
    else:
        return {"error": f"Unknown shape '{shape}'"}
    return {"area": area, "shape": shape}


@tool
def compare_values(a: float, b: float) -> dict:
    """Compare two values. Returns which is larger."""
    if a > b:
        return {"larger": "a", "a": a, "b": b}
    elif b > a:
        return {"larger": "b", "a": a, "b": b}
    return {"larger": "equal", "a": a, "b": b}


@tool
def compute_fibonacci(n: int) -> dict:
    """Compute the nth Fibonacci number (1-indexed, F(1)=1, F(2)=1)."""
    if n <= 0:
        return {"error": "n must be >= 1"}
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return {"fib_value": b if n > 1 else 1, "n": n}


@tool
def check_prime(n: int) -> dict:
    """Check if n is a prime number."""
    if n < 2:
        return {"is_prime": False, "n": n}
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return {"is_prime": False, "n": n}
    return {"is_prime": True, "n": n}


@tool
def json_extract(record_id: str, json_path: str) -> dict:
    """Extract a field from a mock nested record using dot-notation path."""
    mock_records = {
        "R-9912": {"user": {"account": {"balance": 4283.67, "currency": "USD"}, "name": "Vikram"}},
    }
    record = mock_records.get(record_id, {})
    parts = json_path.split(".")
    val: Any = record
    try:
        for p in parts:
            val = val[p]
        return {"value": val, "path": json_path, "record_id": record_id}
    except (KeyError, TypeError):
        return {"error": f"Path '{json_path}' not found in record '{record_id}'"}


@tool
def validate_type(value: Any, expected_type: str) -> dict:
    """Validate that value matches the expected_type (float, int, str, bool)."""
    type_map = {"float": float, "int": int, "str": str, "bool": bool}
    expected = type_map.get(expected_type)
    if not expected:
        return {"error": f"Unknown type '{expected_type}'"}
    valid = isinstance(value, expected)
    positive = (isinstance(value, (int, float)) and value > 0) if expected_type in ("float","int") else True
    return {"validation_passed": valid and positive, "type_found": type(value).__name__}


@tool
def string_strip(text: str) -> dict:
    """Strip leading/trailing whitespace from text."""
    return {"stripped": text.strip()}


@tool
def string_normalize(text: str) -> dict:
    """Lowercase text and remove punctuation."""
    lowered = text.lower()
    cleaned = re.sub(r"[^\w\s]", "", lowered)
    cleaned = " ".join(cleaned.split())
    return {"normalized": cleaned}


@tool
def compute_stats(data: list) -> dict:
    """Compute mean, median, std_dev, min, max of a numeric list."""
    if not data:
        return {"error": "Empty data list"}
    n = len(data)
    mean = sum(data) / n
    sorted_d = sorted(data)
    median = sorted_d[n // 2] if n % 2 else (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in data) / n
    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std_dev": round(math.sqrt(variance), 4),
        "min": min(data),
        "max": max(data),
        "count": n,
    }


@tool
def normalize(value: float, bounds: list) -> dict:
    """Normalize value to [0,1] given [min, max] bounds."""
    lo, hi = bounds[0], bounds[1]
    if hi == lo:
        return {"normalized": 1.0}
    result = min(1.0, max(0.0, (value - lo) / (hi - lo)))
    return {"normalized": round(result, 4)}


@tool
def classify_score(score: float, thresholds: Optional[list] = None) -> dict:
    """Classify a [0,1] score as Low/Medium/High."""
    if thresholds is None:
        thresholds = [0.33, 0.66]
    if score < thresholds[0]:
        cat = "Low"
    elif score < thresholds[1]:
        cat = "Medium"
    else:
        cat = "High"
    return {"classification": cat, "score": score}


@tool
def read_file(file_path: str) -> dict:
    """Read a mock file and return its metadata/contents."""
    meta = _FILES.get(file_path)
    if meta:
        return {"file_path": file_path, "status": "success", **meta}
    return {"error": f"File '{file_path}' not found in mock filesystem."}


@tool
def write_file(file_path: str, content: str) -> dict:
    """Simulate writing content to a file. Always succeeds."""
    return {"file_path": file_path, "bytes_written": len(content), "status": "success"}


@tool
def search_in_text(text: str, search_term: str) -> dict:
    """Count occurrences of search_term in text (case-insensitive)."""
    count = text.lower().count(search_term.lower())
    return {"search_term": search_term, "occurrences": count}


@tool
def filter_data(data: list, condition: dict) -> dict:
    """Filter a list of dicts by condition {field: {op: value}} where op in [gt, lt, eq, gte, lte]."""
    ops = {"gt": lambda a,b: a > b, "lt": lambda a,b: a < b, "eq": lambda a,b: a == b,
           "gte": lambda a,b: a >= b, "lte": lambda a,b: a <= b}
    result = []
    for item in data:
        match = all(
            field in item and op_fn(item[field], val)
            for field, cond in condition.items()
            for op_key, val in cond.items()
            for op_fn in [ops.get(op_key, lambda a,b: False)]
        )
        if match:
            result.append(item)
    return {"filtered": result, "count": len(result)}


@tool
def aggregate_data(data: list, operation: str, field: Optional[str] = None) -> dict:
    """Aggregate list data. operation: sum|count|avg|max|min."""
    if operation == "count":
        return {"result": len(data)}
    if not field:
        return {"error": "field required for sum/avg/max/min"}
    values = [item[field] for item in data if field in item]
    if not values:
        return {"result": 0}
    if operation == "sum":
        return {"result": round(sum(values), 2)}
    elif operation == "avg":
        return {"result": round(sum(values) / len(values), 2)}
    elif operation == "max":
        return {"result": max(values)}
    elif operation == "min":
        return {"result": min(values)}
    return {"error": f"Unknown operation '{operation}'"}


@tool
def merge_datasets(dataset_a: str, dataset_b: str, merge_key: str) -> dict:
    """Merge two named datasets on a key. Returns metadata."""
    meta_a = _FILES.get(dataset_a, {})
    meta_b = _FILES.get(dataset_b, {})
    rows_a = meta_a.get("rows", 0)
    rows_b = meta_b.get("rows", 0)
    merged = rows_a + rows_b
    duplicates = 47 if "users" in dataset_a else 0
    return {"rows_a": rows_a, "rows_b": rows_b, "merged_rows": merged, "duplicate_count": duplicates}


@tool
def deduplicate(data: list, key: Optional[str] = None) -> dict:
    """Remove duplicates from a list or report dedup count from merge metadata."""
    if isinstance(data, dict):
        dups = data.get("duplicate_count", 0)
        final = data.get("merged_rows", 0) - dups
        return {"final_rows": final, "removed": dups}
    seen = []
    unique = []
    for item in data:
        k = item.get(key) if key and isinstance(item, dict) else str(item)
        if k not in seen:
            seen.append(k)
            unique.append(item)
    return {"deduplicated": unique, "count": len(unique), "removed": len(data) - len(unique)}


@tool
def format_output(data: Any, format_type: str = "json") -> dict:
    """Format data as json, csv, markdown, or plain text."""
    if format_type == "json":
        return {"formatted": json.dumps(data, indent=2), "format": "json"}
    elif format_type == "csv":
        if isinstance(data, list) and data and isinstance(data[0], dict):
            headers = ",".join(data[0].keys())
            rows = "\n".join(",".join(str(v) for v in row.values()) for row in data)
            return {"formatted": f"{headers}\n{rows}", "format": "csv"}
        return {"formatted": str(data), "format": "csv"}
    return {"formatted": str(data), "format": format_type}


@tool
def format_summary(data: Any, max_bullets: int = 5) -> dict:
    """Produce a bullet-point summary from a list of strings or dicts."""
    if isinstance(data, list):
        bullets = [f"• {str(item)}" for item in data[:max_bullets]]
    else:
        bullets = [f"• {str(data)}"]
    return {"summary": "\n".join(bullets), "bullet_count": len(bullets)}


@tool
def rank_items(items: list, criterion: str, descending: bool = True) -> dict:
    """Rank a list of dicts by criterion field."""
    try:
        ranked = sorted(items, key=lambda x: x.get(criterion, 0), reverse=descending)
        return {"ranked": ranked, "criterion": criterion}
    except Exception as e:
        return {"error": str(e)}


@tool
def search_database(query: str, source: str = "db_general") -> dict:
    """Mock database search. Returns 3 seeded result snippets."""
    snippets = [
        f"[{source}] Result 1 for '{query}': AI adoption in healthcare reached 34% in 2025 (Source: WHO Report 2025).",
        f"[{source}] Result 2 for '{query}': Market size projected at $45.2B by 2026 with 28% CAGR (MarketsandMarkets).",
        f"[{source}] Result 3 for '{query}': Major barrier remains data privacy regulation (GDPR, HIPAA compliance at 71%).",
    ]
    return {"results": snippets, "source": source, "query": query, "count": 3}


@tool
def extract_facts(text: str) -> dict:
    """Extract key factual claims from text. Returns list of facts."""
    facts = [
        "AI adoption in healthcare reached 34% in 2025",
        "Market projected at $45.2B by 2026",
        "Data privacy is the primary adoption barrier",
    ]
    return {"facts": facts, "count": len(facts)}


@tool
def resolve_contradictions(facts: list) -> dict:
    """Check facts for contradictions and return resolved set."""
    return {"resolved": facts, "contradictions_found": 0, "removed": []}


@tool
def synthesize_content(facts: list, structure: list) -> dict:
    """Synthesize facts into structured sections."""
    sections = {s: f"[Synthesized content for {s} using {len(facts)} facts]" for s in structure}
    return {"sections": sections, "section_count": len(sections)}


@tool
def validate_stats(content: str, sources: list) -> dict:
    """Validate statistics in content against source data."""
    return {"validated_pct": 0.92, "total_stats": 12, "validated": 11, "failed": 1}


@tool
def format_report(data: Any, template: str = "default") -> dict:
    """Format data as a structured report."""
    return {"report": f"[REPORT]\n{json.dumps(data, indent=2, default=str)}", "template": template}


@tool
def validate_report(report: str) -> dict:
    """Validate a report for completeness. Always passes in mock."""
    return {"valid": True, "issues": []}


@tool
def validate_schema(data: Any, schema_name: str = "default") -> dict:
    """Validate data against a named schema."""
    return {"schema_valid": True, "schema": schema_name, "errors": []}


@tool
def query_inventory(category: str) -> dict:
    """Query inventory for a given product category."""
    items = _INVENTORY.get(category, [])
    return {"items": items, "count": len(items), "category": category}


@tool
def lookup_recipe(recipe_id: str) -> dict:
    """Look up a recipe by ID."""
    r = _RECIPES.get(recipe_id)
    return r if r else {"error": f"Recipe '{recipe_id}' not found."}


@tool
def scale_ingredients(ingredients: list, from_servings: int, to_servings: int) -> dict:
    """Scale recipe ingredients from one serving size to another."""
    factor = to_servings / from_servings
    scaled = [{"name": i["name"], "amount": round(i["amount"] * factor, 1), "unit": i["unit"]}
              for i in ingredients]
    return {"scaled": scaled, "factor": factor, "to_servings": to_servings}


@tool
def compute_nutrition(ingredients: list, servings: int) -> dict:
    """Compute approximate nutrition from ingredient list."""
    total_cal = 580 * servings
    total_protein = 18.4 * servings
    return {"total_calories": total_cal, "total_protein_g": total_protein, "servings": servings,
            "protein_per_serving": round(total_protein / servings, 1)}


@tool
def fetch_feed(feed: str, max_items: int = 10) -> dict:
    """Fetch a mock news/data feed."""
    headlines = [
        {"title": "OpenAI releases GPT-5", "category": "AI", "published": "2026-06-27"},
        {"title": "Google Cloud expands Asia", "category": "Cloud", "published": "2026-06-26"},
        {"title": "CISA warns of new ransomware", "category": "Security", "published": "2026-06-27"},
        {"title": "Meta's LLaMA 5 benchmarks", "category": "AI", "published": "2026-06-25"},
        {"title": "AWS re:Invent preview", "category": "Cloud", "published": "2026-06-24"},
        {"title": "Zero-day in Chrome browser", "category": "Security", "published": "2026-06-27"},
        {"title": "Anthropic raises Series E", "category": "AI", "published": "2026-06-23"},
        {"title": "Quantum computing milestone", "category": "Other", "published": "2026-06-22"},
        {"title": "EU AI Act enforcement begins", "category": "AI", "published": "2026-06-21"},
        {"title": "Phishing attacks up 45% YoY", "category": "Security", "published": "2026-06-20"},
    ]
    return {"items": headlines[:max_items], "count": min(max_items, len(headlines)), "feed": feed}


@tool
def classify_text(text: str, categories: list) -> dict:
    """Classify text into one of the given categories."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["ai", "llm", "gpt", "anthropic", "llama", "quantum"]):
        cat = "AI"
    elif any(w in text_lower for w in ["cloud", "aws", "gcp", "azure"]):
        cat = "Cloud"
    elif any(w in text_lower for w in ["security", "ransomware", "phishing", "zero-day", "cisa"]):
        cat = "Security"
    else:
        cat = "Other"
    return {"category": cat, "text": text[:50]}


@tool
def search_flights(origin: str, destination: str, date: Optional[str] = None) -> dict:
    """Mock flight search."""
    return {"cheapest_eur": 680, "duration_hrs": 9.5, "airline": "AirFrance", "stops": 1}


@tool
def search_hotels(city: str, nights: int) -> dict:
    """Mock hotel search."""
    return {"cheapest_per_night_eur": 130, "total_eur": 130 * nights, "hotel": "Ibis Paris"}


@tool
def lookup_visa_fee(nationality: str, destination: str) -> dict:
    """Return visa fee for nationality/destination pair."""
    return {"fee_eur": 80, "validity_days": 90, "type": "Schengen Tourist"}


@tool
def validate_budget(amount: float, budget: float) -> dict:
    """Check if amount exceeds budget."""
    return {"budget_exceeded": amount > budget, "amount": amount, "budget": budget, "surplus": round(budget - amount, 2)}


@tool
def load_calendar(team: str, date: str) -> dict:
    """Load calendar events for a team on a date."""
    events = [
        {"id": f"E{i:02d}", "title": f"Meeting {i}", "start": f"09:{i*5+30:02d}", "end": f"10:{i*5+30:02d}"}
        for i in range(8)
    ]
    return {"events": events, "count": len(events), "team": team, "date": date}


@tool
def detect_conflicts(events: list) -> dict:
    """Detect overlapping events (mock: always finds 3 conflicts)."""
    conflicts = [{"event_a": "E01", "event_b": "E02"}, {"event_a": "E03", "event_b": "E04"},
                 {"event_a": "E05", "event_b": "E06"}]
    return {"conflicts": conflicts, "count": len(conflicts)}


@tool
def propose_slots(conflicts: list) -> dict:
    """Propose alternative time slots for conflicting events."""
    proposals = [{"event": c["event_b"], "proposed_time": f"14:00-15:00"} for c in conflicts]
    return {"proposals": proposals, "count": len(proposals)}


@tool
def validate_schedule(events: list) -> dict:
    """Validate that a schedule has no conflicts."""
    return {"valid": True, "conflicts": 0}


@tool
def format_schedule(events: list) -> dict:
    """Format events as a readable schedule."""
    lines = [f"{e.get('start','?')}-{e.get('end','?')}: {e.get('title','?')}" for e in events]
    return {"schedule": "\n".join(lines), "event_count": len(lines)}


@tool
def score_items(items: list, weights: dict) -> dict:
    """Score items using weighted criteria. items: list of product dicts, weights: {criterion: weight}."""
    scores = {}
    for item in items:
        pid = item.get("id", str(item))
        s = sum(
            weights.get(k, 0) * (item.get(k, 0) / 200.0 if k == "price" else item.get(k, 0) / 5.0)
            for k in weights
        )
        scores[pid] = round(s, 3)
    return {"scores": scores}


@tool
def analyze_complexity(snippet_id: str) -> dict:
    """Analyze cyclomatic complexity of a code snippet."""
    s = _SNIPPETS.get(snippet_id, {})
    return {"complexity_score": s.get("cyclomatic_complexity", 5), "snippet_id": snippet_id}


@tool
def check_naming(snippet_id: str) -> dict:
    """Check naming convention violations in a code snippet."""
    s = _SNIPPETS.get(snippet_id, {})
    return {"violations": s.get("naming_violations", 0)}


@tool
def check_docs(snippet_id: str) -> dict:
    """Check for missing docstrings."""
    s = _SNIPPETS.get(snippet_id, {})
    return {"missing_docs": s.get("missing_docstrings", 0)}


@tool
def detect_duplicates(snippet_id: str) -> dict:
    """Detect duplicate logic patterns."""
    s = _SNIPPETS.get(snippet_id, {})
    return {"duplicate_patterns": s.get("duplicate_patterns", 0)}


@tool
def prioritize_issues(issues: dict) -> dict:
    """Prioritize issues by severity."""
    priority = sorted(issues.items(), key=lambda x: x[1], reverse=True)
    return {"priority_list": [{"issue": k, "score": v} for k, v in priority]}


@tool
def load_expenses(user_id: str, months: int) -> dict:
    """Load expense records for a user over N months."""
    categories = ["food", "transport", "entertainment", "utilities"]
    expenses = {cat: [round(200 + i * 50 + hash(cat) % 100, 2) for i in range(months)] for cat in categories}
    expenses["entertainment"] = [150, 180, 420]  # anomalous spike in month 3
    return {"expenses": expenses, "months": months, "user_id": user_id}


@tool
def categorize_expenses(expenses: dict) -> dict:
    """Return expenses already categorized."""
    return {"categorized": expenses}


@tool
def detect_anomalies(data: Any, threshold: float = 2.0) -> dict:
    """Detect values exceeding threshold * mean in each category."""
    anomalies = []
    if isinstance(data, dict):
        for cat, vals in data.items():
            if isinstance(vals, list) and vals:
                mean = sum(vals) / len(vals)
                for i, v in enumerate(vals):
                    if v > threshold * mean:
                        anomalies.append({"category": cat, "month": i+1, "value": v,
                                          "mean": round(mean,2), "excess_pct": round(v/mean, 2)})
    return {"anomalies": anomalies, "count": len(anomalies)}


@tool
def fetch_evidence(criterion_id: str, project_id: str) -> dict:
    """Fetch evidence document for a compliance criterion."""
    return {"document": f"evidence_{criterion_id}.pdf", "pages": 12, "found": True}


@tool
def extract_passage(document: str, criterion: str) -> dict:
    """Extract relevant passage from a document for a criterion."""
    return {"passage": f"[Excerpt relevant to {criterion} from {document}]", "page": 7}


@tool
def evaluate_criterion(criterion_id: str, passage: str) -> dict:
    """Evaluate a compliance criterion as PASS or FAIL."""
    failing = {"C-004", "C-008", "C-011"}
    result = "FAIL" if criterion_id in failing else "PASS"
    critical = criterion_id == "C-011"
    return {"criterion_id": criterion_id, "result": result, "critical": critical}


@tool
def tally_results(evaluations: list) -> dict:
    """Tally PASS/FAIL counts from evaluation results."""
    pass_count = sum(1 for e in evaluations if e.get("result") == "PASS")
    fail_count = len(evaluations) - pass_count
    critical = sum(1 for e in evaluations if e.get("critical") and e.get("result") == "FAIL")
    return {"pass_count": pass_count, "fail_count": fail_count, "critical_fails": critical}


@tool
def generate_remediation(issues: Any) -> dict:
    """Generate remediation plan for identified issues."""
    return {"remediation": [{"issue": str(i), "action": "Review and correct", "priority": "High"}
                             for i in (issues if isinstance(issues, list) else [issues])]}


@tool
def clean_data(file_path: str) -> dict:
    """Clean a dataset: handle missing values, remove duplicates."""
    meta = _FILES.get(file_path, {"rows": 100})
    return {"cleaned_rows": meta.get("rows", 100), "missing_filled": 12, "duplicates_removed": 3}


@tool
def compute_financial_metrics(data: Any) -> dict:
    """Compute EBITDA, margin, and per-employee metrics from quarterly data."""
    return {
        "EBITDA": {"Q1": 4300000, "Q2": 3300000, "Q3": 4600000, "Q4": 5200000},
        "EBITDA_margin": {"Q1": 0.355, "Q2": 0.289, "Q3": 0.348, "Q4": 0.369},
    }


@tool
def compute_growth_rates(data: Any) -> dict:
    """Compute year-over-year or quarter-over-quarter growth rates."""
    return {"QoQ_growth": {"Q1": None, "Q2": -0.058, "Q3": 0.158, "Q4": 0.068}}


@tool
def identify_extremes(data: dict, metric: str = "EBITDA") -> dict:
    """Identify the quarter with the minimum value for a metric."""
    values = data.get(metric, {})
    if not values:
        return {"min_quarter": "Q2", "min_value": 0}
    min_q = min(values, key=values.get)
    return {"min_quarter": min_q, "min_value": values[min_q]}


@tool
def stress_test(data: dict, reduction: float, quarter: str) -> dict:
    """Apply a revenue reduction to a specific quarter and compute impact."""
    impact = data.get("EBITDA", {}).get(quarter, 3300000) * reduction * -1
    return {"stressed_quarter": quarter, "ebitda_impact": round(impact, 0), "reduction_pct": reduction}


@tool
def run_regression(data: list, periods_ahead: int = 1) -> dict:
    """Run simple linear regression and project future values."""
    if not data or len(data) < 2:
        return {"projection": None}
    n = len(data)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(data) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, data)) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    projection = intercept + slope * (n - 1 + periods_ahead)
    return {"projection": round(projection, 0), "slope": round(slope, 0), "intercept": round(intercept, 0)}


@tool
def compute_compliance_score(evaluations: list) -> dict:
    """Compute overall compliance score as passing_rate."""
    total = len(evaluations)
    passed = sum(1 for e in evaluations if e.get("result") == "PASS")
    return {"overall_compliance_score": round(passed / total, 3) if total else 0.0}


@tool
def run_swot(business_unit: str) -> dict:
    """Run SWOT analysis for a business unit."""
    return {
        "strengths": ["Market leadership", "Strong tech team"],
        "weaknesses": ["High churn rate", "Limited international presence"],
        "opportunities": ["AI platform expansion", "APAC market entry"],
        "threats": ["Increased competition", "Regulatory headwinds"],
        "item_count": 8,
    }


@tool
def benchmark_competitors(competitors: list) -> dict:
    """Benchmark the focal company against competitors."""
    return {"scores": {c: round(0.6 + hash(c) % 40 / 100, 2) for c in competitors}}


@tool
def define_okrs(business_unit: str) -> dict:
    """Define OKRs for a business unit."""
    return {"okrs": [
        {"objective": "Expand platform reach", "key_results": ["10M MAU", "+30% revenue"]},
        {"objective": "Improve retention", "key_results": ["<5% monthly churn", "NPS > 60"]},
        {"objective": "Launch APAC market", "key_results": ["3 partnerships", "$2M ARR"]},
        {"objective": "AI feature parity", "key_results": ["5 AI features", "80% adoption"]},
    ], "count": 4}


@tool
def generate_options(context: Any) -> dict:
    """Generate strategic options."""
    return {"options": ["Platform Expansion", "Market Penetration", "Product Diversification"], "count": 3}


@tool
def evaluate_options(options: list, criteria: list) -> dict:
    """Evaluate strategic options against criteria."""
    scores = {o: round(0.5 + hash(o) % 50 / 100, 2) for o in options}
    return {"scores": scores, "best": max(scores, key=scores.get)}


@tool
def select_option(options: list, scores: dict) -> dict:
    """Select the highest-scoring option with justification."""
    best = max(scores, key=scores.get) if scores else options[0]
    return {"selected": best, "justification": f"{best} scored highest on ROI and market fit."}


@tool
def build_roadmap(option: str, quarters: int) -> dict:
    """Build an implementation roadmap for an option over N quarters."""
    phases = [{"quarter": f"Q{i+1}", "milestone": f"Phase {i+1}: {['Discovery','Build','Launch','Scale'][i]}"} for i in range(quarters)]
    return {"roadmap": phases, "quarters": quarters}


@tool
def compute_resources(roadmap: list) -> dict:
    """Estimate resource requirements for a roadmap."""
    total = len(roadmap) * 600000
    return {"total_investment": total, "headcount_needed": 12, "tech_budget": total * 0.4}


@tool
def compute_roi(investment: float, quarters: int) -> dict:
    """Compute projected ROI over N quarters."""
    roi = round(0.34 * (quarters / 4), 2)
    return {"projected_roi": roi, "payback_quarters": 3}


@tool
def identify_risks(option: str) -> dict:
    """Identify risks and mitigations for a strategic option."""
    return {"risks": [
        {"risk": "Execution delay", "mitigation": "Agile sprints with 2-week reviews", "severity": "Medium"},
        {"risk": "Budget overrun", "mitigation": "Monthly financial reviews", "severity": "High"},
        {"risk": "Talent shortage", "mitigation": "Pre-hire talent pipeline", "severity": "Low"},
    ], "count": 3}


@tool
def validate_financials(data: Any) -> dict:
    """Validate that financial projections are internally consistent."""
    return {"consistent": True, "issues": []}


@tool
def format_deck(data: Any) -> dict:
    """Format data as a strategy deck structure."""
    return {"slides": 12, "format": "PowerPoint", "written": True}


@tool
def load_performance_data(business_unit: str) -> dict:
    """Load current performance data for a business unit."""
    return {"revenue": 8200000, "churn": 0.072, "nps": 52, "mau": 4200000}


@tool
def extract_competitor_data(competitor: str, source: str) -> dict:
    """Extract competitor data from a source."""
    return {"competitor": competitor, "source": source, "pricing": 99, "market_share": 0.12,
            "features": ["Feature A", "Feature B"], "recent_news": f"{competitor} raised Series C"}


@tool
def normalize_metrics(data: list) -> dict:
    """Normalize all numeric metrics to [0,1] scale."""
    return {"normalized": data, "status": "normalized"}


@tool
def compute_position_matrix(competitors: list) -> dict:
    """Compute competitive position matrix."""
    return {"matrix": {c: {"x": 0.5, "y": 0.5} for c in competitors}, "quadrants": 4}


@tool
def identify_gaps(matrix: dict, focal: str) -> dict:
    """Identify top gaps versus focal competitor."""
    return {"gaps": ["Pricing", "Mobile UX", "API ecosystem", "Support SLA"], "count": 4}


@tool
def generate_recommendations(gaps: list, constraints: Any = None) -> dict:
    """Generate strategic recommendations to close gaps."""
    recs = [{"gap": g, "recommendation": f"Invest in {g} improvement", "priority": i+1}
            for i, g in enumerate(gaps[:5])]
    return {"recommendations": recs, "count": len(recs)}


@tool
def load_constraints(file_path: str) -> dict:
    """Load business constraints from a file."""
    return {"constraints": ["Budget cap $3M", "No acquisitions FY2026", "Headcount freeze Q1"], "count": 3}


@tool
def validate_recommendations(recommendations: list, constraints: list) -> dict:
    """Check recommendations against constraints."""
    conflicts = [{"rec": recommendations[0]["recommendation"] if recommendations else "", "constraint": "Budget cap $3M"}]
    return {"conflicts": conflicts, "conflict_count": len(conflicts), "valid_recs": len(recommendations) - len(conflicts)}


@tool
def compute_probability(data: Any) -> dict:
    """Compute anomaly or event probability from signal data."""
    return {"probability": 0.87, "confidence": 0.91}


@tool
def apply_fft(signal_data: Any) -> dict:
    """Apply Fast Fourier Transform to signal data."""
    return {"fft_output": "complex_array", "length": 8192, "sample_rate": 1000}


@tool
def extract_frequencies(fft_output: Any) -> dict:
    """Extract dominant frequencies from FFT output."""
    return {"dominant_freq_hz": 3.7, "secondary_freqs": [7.4, 11.1], "peak_amplitude": 0.847}


@tool
def bandpass_filter(signal: Any, low_hz: float = 1.0, high_hz: float = 10.0) -> dict:
    """Apply bandpass filter to a signal."""
    return {"filtered": True, "passband": f"{low_hz}-{high_hz}Hz", "attenuation_db": 40}


@tool
def compute_snr(signal: Any) -> dict:
    """Compute Signal-to-Noise Ratio."""
    return {"snr_db": 18.4, "noise_floor_db": -42.1}


@tool
def detect_peaks(signal: Any, threshold: float = 0.5) -> dict:
    """Detect anomaly peaks in a signal."""
    return {"peaks": [{"time": 1.2, "amplitude": 0.91}, {"time": 3.7, "amplitude": 0.88},
                      {"time": 6.1, "amplitude": 0.95}], "count": 3}


@tool
def classify_peaks(peaks: list) -> dict:
    """Classify detected peaks by type."""
    types = ["transient", "periodic", "noise"]
    classified = [{"peak": p, "type": types[i % 3]} for i, p in enumerate(peaks)]
    return {"classified": classified, "count": len(classified)}


@tool
def correlate_events(peaks: list, event_log: str) -> dict:
    """Correlate signal peaks with system events."""
    return {"correlated": [{"peak": peaks[0], "event": "deploy_v2.3.1"}], "correlation_count": 1}


@tool
def parse_logs(service: str, hours: int) -> dict:
    """Parse system logs for a service over N hours."""
    return {"entries_parsed": 142847, "error_count": 2847, "warn_count": 5123,
            "services": [service], "time_range_hrs": hours}


@tool
def root_cause_analysis(patterns: list) -> dict:
    """Identify root causes from anomaly patterns."""
    return {"root_causes": ["DB connection pool exhaustion", "Memory leak in auth service"],
            "primary_cause": "DB connection pool exhaustion"}


@tool
def simulate_fix(root_cause: str) -> dict:
    """Simulate applying a fix for a root cause."""
    return {"fix_applied": root_cause, "simulated": True, "expected_improvement": 0.73}


@tool
def verify_fix(simulation: dict) -> dict:
    """Verify that the simulated fix actually reduces errors."""
    return {"verified": True, "error_reduction_pct": simulation.get("expected_improvement", 0.7)}


@tool
def acknowledge_alert(incident_id: str) -> dict:
    """Acknowledge an alert/incident."""
    return {"acknowledged": True, "incident_id": incident_id, "acknowledged_by": "agent"}


@tool
def triage_incident(incident_id: str) -> dict:
    """Triage incident severity."""
    return {"severity": "P1", "category": "availability", "affected_component": "payment-service"}


@tool
def page_oncall(team: str, severity: str) -> dict:
    """Page on-call team members."""
    return {"paged": True, "team": team, "responders": ["sre-lead", "backend-lead"]}


@tool
def investigate_logs(service: str, incident_id: str) -> dict:
    """Investigate service logs for incident root cause."""
    return {"root_cause": "DB connection pool exhaustion", "first_error_at": "2026-03-14T02:14:33Z"}


@tool
def identify_affected(service: str) -> dict:
    """Identify services affected by an incident."""
    return {"affected": ["payment-service", "checkout-api", "notification-service"], "count": 3}


@tool
def estimate_impact(affected: list) -> dict:
    """Estimate user impact of an incident."""
    return {"affected_users": 47832, "revenue_at_risk_per_min": 2340, "sla_breach": True}


@tool
def apply_mitigation(root_cause: str) -> dict:
    """Apply mitigation for a root cause."""
    return {"mitigation": "Increased DB connection pool from 20 to 100", "applied": True}


@tool
def verify_mitigation(mitigation: dict) -> dict:
    """Verify that mitigation was successful."""
    return {"success": True, "error_rate_before": 0.34, "error_rate_after": 0.02}


@tool
def restore_service(service: str) -> dict:
    """Restore a degraded service to normal operation."""
    return {"service": service, "restored": True, "downtime_minutes": 47}


@tool
def conduct_postmortem(incident_id: str) -> dict:
    """Generate postmortem analysis for an incident."""
    return {"timeline": "02:14-03:01 UTC", "contributing_factors": 3, "action_items_count": 5}


@tool
def extract_action_items(postmortem: dict) -> dict:
    """Extract action items from a postmortem."""
    return {"action_items": [
        "Increase connection pool sizing (owner: DBA, due: 2026-07-05)",
        "Add connection pool metrics to Grafana dashboard",
        "Create runbook for pool exhaustion",
        "Implement circuit breaker for DB",
        "Load test payment service quarterly",
    ], "count": 5}


@tool
def create_jira_tickets(items: list) -> dict:
    """Create Jira tickets for action items."""
    tickets = [f"INC-{2000+i}" for i in range(len(items))]
    return {"created": tickets, "count": len(tickets)}


@tool
def lookup_employee(employee_id: str) -> dict:
    """Look up an employee record."""
    return {"employee_id": employee_id, "name": "Rohan Mehta", "role": "Senior Engineer",
            "department": "Platform", "start_date": "2026-07-01"}


@tool
def determine_permissions(role: str) -> dict:
    """Determine IAM permissions for a role."""
    return {"policies": [f"POL-{i:03d}" for i in range(1, 9)], "count": 8, "role": role}


@tool
def create_iam_account(employee_id: str, role: str) -> dict:
    """Create IAM account for employee."""
    return {"account_id": f"IAM-{employee_id}", "created": True, "role": role}


@tool
def assign_policies(account_id: str, policies: list) -> dict:
    """Assign IAM policies to an account."""
    return {"assigned": policies, "count": len(policies), "account_id": account_id}


@tool
def provision_licenses(employee_id: str, tools: list) -> dict:
    """Provision software licenses for an employee."""
    default_tools = ["GitHub Enterprise", "Jira", "Confluence", "Slack Pro", "DataDog"]
    t = tools[:5] if tools else default_tools
    return {"provisioned": t, "count": len(t)}


@tool
def create_email(employee_id: str, name: str) -> dict:
    """Create a corporate email account."""
    username = name.lower().replace(" ", ".")
    return {"email": f"{username}@company.com", "created": True}


@tool
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    return {"sent": True, "to": to, "subject": subject, "message_id": "MSG-20260701-001"}


@tool
def update_org_chart(employee_id: str, manager: str, department: str) -> dict:
    """Add employee to org chart."""
    return {"updated": True, "employee_id": employee_id, "reports_to": manager}


@tool
def schedule_meetings(employee_id: str, meeting_types: list) -> dict:
    """Schedule onboarding meetings for a new employee."""
    meetings = [{"type": m, "date": "2026-07-03", "attendees": 3} for m in (meeting_types or ["HR Intro", "Team Sync", "1:1 with Manager"])]
    return {"scheduled": meetings, "count": len(meetings)}


@tool
def create_jira_ticket(project: str, summary: str, issue_type: str = "Task") -> dict:
    """Create a single Jira ticket."""
    return {"ticket_id": f"{project}-{hash(summary) % 9999:04d}", "created": True, "summary": summary}


@tool
def audit_actions(actions: list) -> dict:
    """Audit a list of onboarding actions for completeness."""
    return {"audit_passed": True, "checked": len(actions), "issues": []}


@tool
def validate_checklist(checklist: dict) -> dict:
    """Validate that all checklist items are complete."""
    total = len(checklist)
    passed = sum(1 for v in checklist.values() if v)
    return {"all_passed": passed == total, "passed": passed, "total": total}


@tool
def query_database(db_name: str, query: str) -> dict:
    """Run a query against a mock database."""
    row_counts = {"db_legacy": 284750, "db_new": 284750}
    count = row_counts.get(db_name, 10000)
    return {"rows": count, "db": db_name, "query": query}


@tool
def compare_samples(table: str, sample_size: int = 100) -> dict:
    """Compare random samples between legacy and new databases."""
    return {"sample_size": sample_size, "match_rate": 0.99, "mismatches": 1}


@tool
def validate_foreign_keys(table: str, db_name: str) -> dict:
    """Validate all foreign key constraints."""
    return {"valid": True, "fk_count": 12, "violations": 0}


@tool
def check_indexes(table: str, db_name: str) -> dict:
    """Check that all required indexes exist."""
    return {"indexes_found": 8, "indexes_required": 8, "missing": []}


@tool
def check_triggers(table: str, db_name: str) -> dict:
    """Check that triggers have been migrated."""
    return {"triggers_migrated": 3, "total": 3, "missing": []}


@tool
def validate_procedures(db_name: str) -> dict:
    """Validate stored procedures in a database."""
    return {"procedures_valid": 5, "total": 5, "errors": []}


@tool
def run_equivalence_queries(table: str, count: int = 10) -> dict:
    """Run equivalence queries between old and new DB."""
    return {"queries_run": count, "passed": count - 1, "failed": 1, "discrepancies": [{"row_id": 284319}]}


@tool
def compute_checksum(table: str, db_name: str) -> dict:
    """Compute checksum over entire table."""
    checksums = {"db_legacy": "a3f2c891", "db_new": "a3f2c891"}
    c = checksums.get(db_name, "unknown")
    return {"checksum": c, "match": True}


@tool
def generate_migration_report(data: Any) -> dict:
    """Generate a migration validation report."""
    return {"report": "[Migration Report]", "written": True, "discrepancies": 2}


@tool
def profile_data(dataset: str) -> dict:
    """Profile a dataset: shape, types, missing values."""
    return {"rows": 8420, "columns": 24, "missing_pct": 0.034, "numeric_cols": 18, "categorical_cols": 6}


@tool
def handle_missing(data: dict) -> dict:
    """Handle missing values in a dataset."""
    return {"rows_before": data.get("rows", 8420), "rows_after": 8420, "imputed": 287}


@tool
def encode_features(data: dict) -> dict:
    """Encode categorical features."""
    return {"categorical_encoded": 6, "total_features": 24, "new_feature_count": 42}


@tool
def split_dataset(data: dict, split: list) -> dict:
    """Split dataset into train/val/test."""
    total = data.get("rows", 8420)
    return {
        "train": int(total * split[0]),
        "val": int(total * split[1]),
        "test": int(total * split[2]),
    }


@tool
def train_model(model_type: str, data: dict) -> dict:
    """Train a classification model."""
    perf = {"LogisticRegression": {"acc": 0.782, "f1": 0.761, "auc": 0.801},
            "RandomForest": {"acc": 0.823, "f1": 0.811, "auc": 0.834},
            "XGBoost": {"acc": 0.841, "f1": 0.829, "auc": 0.847}}
    return {"model": model_type, **perf.get(model_type, {"acc": 0.75, "f1": 0.73, "auc": 0.78})}


@tool
def evaluate_models(models: list, data: dict) -> dict:
    """Evaluate multiple models and return comparison."""
    return {"comparison": {m["model"]: m["auc"] for m in models}, "best": "XGBoost"}


@tool
def select_best_model(comparison: dict) -> dict:
    """Select the best-performing model."""
    best = max(comparison, key=comparison.get)
    return {"best_model": best, "auc": comparison[best]}


@tool
def tune_hyperparameters(model_type: str, data: dict) -> dict:
    """Tune hyperparameters for a model."""
    return {"tuned_model": model_type, "tuned_auc": 0.847, "best_params": {"n_estimators": 200, "max_depth": 6}}


@tool
def evaluate_model(model: dict, data: dict) -> dict:
    """Evaluate a single model on test set."""
    return {"model": model.get("tuned_model", "XGBoost"), "test_auc": 0.831, "test_acc": 0.838}


@tool
def write_model_card(model: dict, metrics: dict) -> dict:
    """Write a model card document."""
    return {"written": True, "path": "results/model_card.md"}


@tool
def load_criteria(standard: str, batch: int = 1) -> dict:
    """Load compliance criteria for a standard."""
    criteria = [f"C-{i:03d}" for i in range(1, 13)]
    return {"criteria": criteria, "count": len(criteria), "standard": standard}


@tool
def synthesize_graph(entities: Any, relations: Any) -> dict:
    """Synthesize a knowledge graph from entities and relations."""
    return {"nodes": 87, "edges": 134, "density": round(134 / (87 * 86), 4)}


@tool
def extract_entities(text: str) -> dict:
    """Extract named entities from text."""
    return {"entities": [{"name": "BERT", "type": "Model"}, {"name": "Google", "type": "Org"},
                         {"name": "Attention Mechanism", "type": "Concept"}], "count": 87}


@tool
def classify_entities(entities: list) -> dict:
    """Classify entity types."""
    return {"classified": entities, "types": {"Model": 23, "Org": 18, "Person": 15, "Concept": 31}}


@tool
def extract_relations(entities: list) -> dict:
    """Extract relations between entities."""
    return {"relations": [{"from": "Google", "rel": "developed", "to": "BERT"}], "count": 134}


@tool
def resolve_coreferences(text: str) -> dict:
    """Resolve co-references in text."""
    return {"resolved": True, "coreference_chains": 12}


@tool
def validate_ontology(entities: list, ontology: str) -> dict:
    """Validate entities against an ontology."""
    return {"valid": True, "coverage": 0.89, "unmapped": 3}


@tool
def compute_graph_metrics(nodes: int, edges: int) -> dict:
    """Compute graph density and other metrics."""
    density = round(edges / (nodes * (nodes - 1)) if nodes > 1 else 0, 4)
    return {"nodes": nodes, "edges": edges, "density": density}


@tool
def detect_clusters(graph: Any) -> dict:
    """Detect disconnected clusters in a graph."""
    return {"clusters": 3, "cluster_sizes": [45, 31, 11]}


@tool
def enrich_entities(entities: list, kb: str = "wikidata") -> dict:
    """Enrich entities with external knowledge base lookups."""
    return {"enriched_count": len(entities), "kb": kb, "enrichment_rate": 0.87}


@tool
def serialize_graph(graph: Any, format_type: str = "json-ld") -> dict:
    """Serialize graph to JSON-LD or other format."""
    return {"serialized": True, "format": format_type, "output_path": "results/graph.jsonld"}


@tool
def build_scenario(name: str, base_data: Any, adjustment: float) -> dict:
    """Build a financial scenario with revenue adjustment."""
    return {"scenario": name, "adjustment": adjustment, "revenue_factor": 1 + adjustment}


@tool
def run_dcf(scenario: dict, discount_rate: float, terminal_growth: float) -> dict:
    """Run DCF valuation for a scenario."""
    base_equity = {"base": 182000000, "bear": 98000000, "bull": 247000000}
    name = scenario.get("scenario", "base")
    return {"equity_value": base_equity.get(name, 150000000), "scenario": name}


@tool
def compute_sensitivity(scenarios: list) -> dict:
    """Compute equity value sensitivity across scenarios."""
    values = [s.get("equity_value", 0) for s in scenarios]
    return {"min_value": min(values), "max_value": max(values), "range": max(values) - min(values)}


@tool
def identify_breakeven(data: Any) -> dict:
    """Identify break-even revenue conditions."""
    return {"break_even_revenue": 14200000, "break_even_margin": 0.08}


@tool
def create_outline(topic: str, sections: int = 5) -> dict:
    """Create a structured outline for a content piece."""
    section_names = ["Introduction", "Background", "Core Concepts", "Applications", "Conclusion"]
    return {"outline": section_names[:sections], "section_count": sections}


@tool
def draft_section(section_name: str, topic: str, target_words: int = 200) -> dict:
    """Draft a content section."""
    return {"section": section_name, "content": f"[Draft of {section_name} for {topic}... ~{target_words} words]",
            "word_count": target_words}


@tool
def fact_check(content: str) -> dict:
    """Fact-check content and return verification results."""
    return {"claims_found": 15, "verified": 14, "unverified": 1, "pass_rate": 0.933}


@tool
def fix_inaccuracies(content: str, issues: list) -> dict:
    """Fix identified inaccuracies in content."""
    return {"fixed": len(issues), "updated_content": content}


@tool
def check_readability(text: str) -> dict:
    """Check the Flesch-Kincaid reading grade level of text."""
    return {"grade_level": 10, "flesch_score": 68.4, "reading_ease": "Standard"}


@tool
def adjust_complexity(text: str, target_grade: int) -> dict:
    """Adjust text complexity to target grade level."""
    return {"adjusted": True, "new_grade": target_grade, "changes_made": 8}


@tool
def seo_analyze(text: str) -> dict:
    """Analyze text for SEO metrics."""
    return {"seo_score": 0.78, "keyword_density": 0.021, "readability_ok": True}


@tool
def format_html(content: dict) -> dict:
    """Format content sections as an HTML article."""
    return {"html": "<article>...</article>", "valid": True, "byte_size": 4820}


@tool
def validate_html(html: str) -> dict:
    """Validate HTML for correctness."""
    return {"valid": True, "errors": [], "warnings": 2}


# ─── Tool registry ─────────────────────────────────────────────────────────────

_ALL_TOOLS = [
    convert_units, calculate, validate_range, classify_bmi, parse_date, date_diff,
    lookup_product, lookup_user, lookup_client, lookup_order, lookup_address,
    lookup_customer, lookup_timezone, convert_timezone, format_datetime,
    format_currency, convert_currency, classify_temperature, sort_list, slice_list,
    count_words, compute_word_frequency, calculate_area, compare_values,
    compute_fibonacci, check_prime, json_extract, validate_type, string_strip,
    string_normalize, search_in_text, filter_data, aggregate_data, merge_datasets,
    deduplicate, format_output, format_summary, rank_items, search_database,
    extract_facts, resolve_contradictions, synthesize_content, validate_stats,
    format_report, validate_report, validate_schema, query_inventory, lookup_recipe,
    scale_ingredients, compute_nutrition, fetch_feed, classify_text, search_flights,
    search_hotels, lookup_visa_fee, validate_budget, load_calendar, detect_conflicts,
    propose_slots, validate_schedule, format_schedule, score_items, analyze_complexity,
    check_naming, check_docs, detect_duplicates, prioritize_issues, load_expenses,
    categorize_expenses, detect_anomalies, fetch_evidence, extract_passage,
    evaluate_criterion, tally_results, generate_remediation, clean_data,
    compute_financial_metrics, compute_growth_rates, identify_extremes, stress_test,
    run_regression, compute_compliance_score, run_swot, benchmark_competitors,
    define_okrs, generate_options, evaluate_options, select_option, build_roadmap,
    compute_resources, compute_roi, identify_risks, validate_financials, format_deck,
    load_performance_data, extract_competitor_data, normalize_metrics,
    compute_position_matrix, identify_gaps, generate_recommendations, load_constraints,
    validate_recommendations, compute_probability, apply_fft, extract_frequencies,
    bandpass_filter, compute_snr, detect_peaks, classify_peaks, correlate_events,
    parse_logs, root_cause_analysis, simulate_fix, verify_fix, acknowledge_alert,
    triage_incident, page_oncall, investigate_logs, identify_affected, estimate_impact,
    apply_mitigation, verify_mitigation, restore_service, conduct_postmortem,
    extract_action_items, create_jira_tickets, lookup_employee, determine_permissions,
    create_iam_account, assign_policies, provision_licenses, create_email, send_email,
    update_org_chart, schedule_meetings, create_jira_ticket, audit_actions,
    validate_checklist, query_database, compare_samples, validate_foreign_keys,
    check_indexes, check_triggers, validate_procedures, run_equivalence_queries,
    compute_checksum, generate_migration_report, profile_data, handle_missing,
    encode_features, split_dataset, train_model, evaluate_models, select_best_model,
    tune_hyperparameters, evaluate_model, write_model_card, load_criteria,
    extract_entities, classify_entities, extract_relations, resolve_coreferences,
    validate_ontology, compute_graph_metrics, detect_clusters, enrich_entities,
    serialize_graph, build_scenario, run_dcf, compute_sensitivity, identify_breakeven,
    create_outline, draft_section, fact_check, fix_inaccuracies, check_readability,
    adjust_complexity, seo_analyze, format_html, validate_html, compute_stats,
    normalize, classify_score, read_file, write_file, synthesize_graph,
]

_TOOL_MAP = {t.name: t for t in _ALL_TOOLS}


def get_tools_for_task(task: dict) -> list:
    """Return the tool objects required by a task (falls back to full tool set)."""
    required = task.get("tools_required", [])
    if not required:
        return _ALL_TOOLS
    tools = []
    seen = set()
    for name in required:
        if name in _TOOL_MAP and name not in seen:
            tools.append(_TOOL_MAP[name])
            seen.add(name)
    # Always include at least the required tools; if not found fall back to all
    return tools if tools else _ALL_TOOLS


def get_all_tools() -> list:
    return _ALL_TOOLS
