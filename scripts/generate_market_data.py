import struct
import random
import time
import os

# Configuration
NUM_ORDERS = 1_000_000
OUTPUT_FILE = "../benchmarks/market_data.bin"

print(f"[*] Generating {NUM_ORDERS} random orders...")

with open(OUTPUT_FILE, "wb") as f:
    for i in range(NUM_ORDERS):
        # Generate raw binary data matching the C++ struct alignment
        # OrderId(8), Price(4), Qty(4), Side(1), Type(1), Padding(6), Timestamp(8)
        
        order_id = i
        price = random.randint(100, 500)
        qty = random.randint(1, 100)
        side = random.choice([0, 1]) # 0=Buy, 1=Sell
        order_type = 0 if random.random() > 0.2 else 1 # 80% Limit, 20% Market
        timestamp = int(time.time_ns())
        
        # Packing struct: Q I I B B xxxxxxx Q
        data = struct.pack("=QIIBB6xQ", order_id, price, qty, side, order_type, timestamp)
        f.write(data)

print(f"[+] Successfully generated {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB of market data.")