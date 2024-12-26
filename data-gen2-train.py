import argparse
import random
import string
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

# ----------------------------
# Single-chunk generation
# ----------------------------
def generate_chunk(chunk_id, chunk_size, seed, start_date, end_date, category_list, categories,
                   payment_methods, user_devices, user_cities, user_genders):
    """
    Generates a chunk (subset) of the synthetic e-commerce data for customer segmentation.
    Returns a Pandas DataFrame.
    """

    random.seed(seed + chunk_id)
    np.random.seed(seed + chunk_id)

    transaction_ids = []
    user_ids = []
    product_ids = []
    category_col = []
    subcategory_col = []
    prices = []
    quantities = []
    total_amounts = []
    payment_col = []
    timestamps = []
    devices = []
    cities = []
    ages = []
    genders = []
    incomes = []
    loyalty_scores = []
    purchase_frequencies = []
    avg_spend_per_transaction = []
    preferred_categories = []
    return_probabilities = []
    return_labels = []  # New: Product return labels
    default_labels = []  # New: Payment default labels

    delta = end_date - start_date
    for i in range(chunk_size):
        global_index = chunk_id * chunk_size + i + 1

        transaction_ids.append(global_index)

        user_id = random.randint(1, chunk_size // 10)
        user_ids.append(user_id)

        cat = random.choice(category_list)
        subcat = random.choice(categories[cat])
        category_col.append(cat)
        subcategory_col.append(subcat)

        product_ids.append("PID_" + "".join(random.choices(string.digits, k=8)))

        price = round(random.uniform(1.0, 999.99), 2)
        prices.append(price)

        qty = random.randint(1, 5)
        quantities.append(qty)

        total_amount = round(price * qty, 2)
        total_amounts.append(total_amount)

        payment_col.append(random.choice(payment_methods))

        random_second = random.randint(0, int(delta.total_seconds()))
        tx_time = start_date + timedelta(seconds=random_second)
        timestamps.append(tx_time.strftime("%Y-%m-%d %H:%M:%S"))

        devices.append(random.choice(user_devices))
        cities.append(random.choice(user_cities))
        ages.append(random.randint(18, 70))
        genders.append(random.choice(user_genders))
        incomes.append(random.randint(20000, 200000))

        # Additional Features for Segmentation
        loyalty_scores.append(round(np.random.uniform(1, 10), 2))  # Loyalty score (1-10 scale)
        purchase_frequencies.append(np.random.poisson(12))  # Average number of transactions per month
        avg_spend_per_transaction.append(round(np.random.uniform(20, 500), 2))  # Avg. spend
        preferred_categories.append(cat)  # Assign preferred category based on current transaction
        return_probabilities.append(round(np.random.uniform(0, 1), 2))  # Probability of return

        # Logic for Return Label (Higher chance of return for high price or high quantity)
        if price > 500 or qty > 3:
            return_labels.append(1)  # Returned
        else:
            return_labels.append(0)  # Not returned

        # Logic for Default Label (Higher chance of default for low income or high total amount)
        if incomes[-1] < 50000 and total_amount > 500:
            default_labels.append(1)  # Defaulted
        else:
            default_labels.append(0)  # Not defaulted

    data = {
        "transaction_id": transaction_ids,
        "user_id": user_ids,
        "product_id": product_ids,
        "category": category_col,
        "sub_category": subcategory_col,
        "price": prices,
        "quantity": quantities,
        "total_amount": total_amounts,
        "payment_method": payment_col,
        "timestamp": timestamps,
        "user_device": devices,
        "user_city": cities,
        "user_age": ages,
        "user_gender": genders,
        "user_income": incomes,
        "loyalty_score": loyalty_scores,
        "purchase_frequency": purchase_frequencies,
        "avg_spend_per_transaction": avg_spend_per_transaction,
        "preferred_category": preferred_categories,
        "return_probability": return_probabilities,
        "return_label": return_labels,  # New Column
        "default_label": default_labels,  # New Column
    }

    return pd.DataFrame(data)


def generate_ecommerce_data_mp(num_rows=1_000_000, seed=42, n_cores=None):
    """
    Generate synthetic e-commerce data using multiprocessing.
    :param num_rows: Total number of rows/transactions.
    :param seed: Base random seed.
    :param n_cores: Number of processes to spawn. Defaults to cpu_count() if None.
    :return: Pandas DataFrame containing all generated data.
    """

    # If n_cores not specified, use all available CPU cores
    if n_cores is None:
        n_cores = cpu_count()

    # Basic parameters for data generation
    categories = {
        "Electronics": ["Smartphones", "Laptops", "Headphones", "Wearables"],
        "Clothing": ["Men_Shirts", "Women_Dresses", "Shoes", "Accessories"],
        "Home": ["Furniture", "Kitchen", "Decor", "Garden"],
        "Beauty": ["Skincare", "Makeup", "Fragrance", "Haircare"],
        "Sports": ["Gym_Equipment", "Outdoor", "Sportswear", "Footwear"],
        "Books": ["Fiction", "NonFiction", "Comics", "Textbooks"],
        "Automotive": ["Car_Accessories", "Motorcycle_Parts", "Car_Electronics"],
    }
    category_list = list(categories.keys())

    payment_methods = ["CreditCard", "DebitCard", "PayPal", "CashOnDelivery"]
    user_devices = ["Desktop", "Mobile", "Tablet"]
    user_genders = ["M", "F", "Other"]
    user_cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "San Antonio", "San Diego", "Dallas", "San Jose", "Austin"
    ]

    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    # Determine chunk size for each process
    chunk_size = num_rows // n_cores
    remainder = num_rows % n_cores

    # Prepare list of (chunk_id, chunk_size) to feed each process
    tasks = []
    for i in range(n_cores):
        # Distribute remainder among first few chunks
        size = chunk_size + (1 if i < remainder else 0)

        tasks.append(
            (i, size, seed, start_date, end_date, category_list, categories,
             payment_methods, user_devices, user_cities, user_genders)
        )

    # Use a Pool of processes to generate chunks in parallel
    with Pool(processes=n_cores) as pool:
        df_chunks = pool.starmap(generate_chunk, tasks)

    # Concatenate chunks
    df_final = pd.concat(df_chunks, ignore_index=True)
    return df_final


def main():
    import sys
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce CSV data in parallel.")
    parser.add_argument("--rows", type=int, default=1_000_000,
                        help="Number of rows to generate.")
    parser.add_argument("--output", type=str, default="ecommerce_data.csv",
                        help="Output CSV file name.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--cores", type=int, default=None,
                        help="Number of CPU cores to use (default: all).")

    args = parser.parse_args()

    print(f"Generating {args.rows} rows with multiprocessing...")
    print(f"Using {args.cores if args.cores else cpu_count()} core(s).")

    start_time = time.time()
    df = generate_ecommerce_data_mp(
        num_rows=args.rows,
        seed=args.seed,
        n_cores=args.cores
    )

    # Save to CSV
    df.to_csv(args.output, index=False)
    elapsed = time.time() - start_time

    print(f"Data generation complete. Saved to {args.output}")
    print(f"Time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    # On Windows, multiprocessing requires this guard
    main()
