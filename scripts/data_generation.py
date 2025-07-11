from faker import Faker
import pandas as pd
import random

def generate_support_tickets(num_tickets=1000):
    fake = Faker()
    categories = [
        'Billing Issue', 
        'Technical Problem', 
        'Account Access', 
        'Product Inquiry', 
        'Refund Request', 
        'Shipping Concern', 
        'Service Complaint'
    ]
    
    tickets = []
    for _ in range(num_tickets):
        num_labels = random.randint(1, 3)
        ticket_categories = random.sample(categories, num_labels)
        ticket = {
            'text': fake.text(),
            'categories': ticket_categories
        }
        tickets.append(ticket)
    
    return pd.DataFrame(tickets)

df = generate_support_tickets()
df.to_csv('../data/raw/support_tickets.csv', index=False)
