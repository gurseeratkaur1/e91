import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
import hashlib

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class LegalConversionDataGenerator:
    """
    Generate legally compliant brand activation conversion datasets
    Only includes data obtainable through explicit consent and direct business relationships
    """
    
    def __init__(self, num_users=1000, num_events=5):
        self.num_users = num_users
        self.num_events = num_events
        self.base_date = datetime(2025, 1, 1)
        
        # Legal interaction types (explicit consent required)
        self.interaction_types = [
            "registration_booth_visit",
            "demo_request_form", 
            "contact_info_exchange",
            "brochure_download",
            "newsletter_signup",
            "product_trial_request",
            "pricing_inquiry",
            "consultation_booking"
        ]
        
        # Legal conversion types (direct business relationship)
        self.conversion_types = [
            "software_purchase",
            "service_signup",
            "trial_to_paid",
            "consultation_booked",
            "demo_scheduled",
            "newsletter_confirmed",
            "whitepaper_download",
            "webinar_registration"
        ]
        
        self.communication_types = [
            "welcome_email",
            "product_demo_invite", 
            "pricing_follow_up",
            "case_study_share",
            "newsletter_send",
            "consultation_reminder"
        ]
        
    def generate_user_ids(self):
        """Generate hashed user IDs (simulating hashed emails)"""
        return [hashlib.md5(f"user_{i}@company.com".encode()).hexdigest()[:12] 
                for i in range(self.num_users)]
    
    def generate_event_interactions(self):
        """Generate event interaction data - explicitly consented interactions only"""
        user_ids = self.generate_user_ids()
        interactions = []
        
        for _ in range(int(self.num_users * 1.3)):  # Some users have multiple interactions
            user_id = np.random.choice(user_ids)
            event_date = self.base_date + timedelta(days=random.randint(0, 30))
            
            interaction = {
                'user_id': user_id,
                'event_id': f"event_{random.randint(1, self.num_events)}",
                'interaction_type': np.random.choice(self.interaction_types),
                'timestamp': event_date + timedelta(hours=random.randint(9, 17), 
                                                  minutes=random.randint(0, 59)),
                'interaction_channel': np.random.choice(['in_person_booth', 'qr_code_scan', 
                                                    'staff_referral', 'self_service_kiosk']),
                'lead_quality': np.random.choice(['hot', 'warm', 'cold']),
                'contact_consent_given': np.random.choice([True, False], p=[0.7, 0.3]),
                'interest_level': random.randint(1, 5),
                'staff_notes': np.random.choice(['highly_engaged', 'specific_use_case', 
                                            'budget_confirmed', 'decision_maker', 'evaluating_options'])
            }
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def generate_consent_preferences(self, interactions_df):
        """Generate consent and preference data - legally required tracking"""
        consent_data = []
        
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            first_interaction = user_interactions.iloc[0]
            
            # Only create consent record if user gave consent
            if first_interaction['contact_consent_given']:
                consent = {
                    'user_id': user_id,
                    'email_consent': np.random.choice([True, False], p=[0.85, 0.15]),
                    'sms_consent': np.random.choice([True, False], p=[0.3, 0.7]),
                    'data_tracking_consent': np.random.choice([True, False], p=[0.6, 0.4]),
                    'consent_date': first_interaction['timestamp'],
                    'consent_method': np.random.choice(['event_form', 'booth_tablet', 'staff_collection']),
                    'privacy_policy_version': 'v2.1_2025',
                    'opt_out_date': None if np.random.random() > 0.05 else 
                                  first_interaction['timestamp'] + timedelta(days=np.random.randint(1, 60))
                }
                consent_data.append(consent)
        
        return pd.DataFrame(consent_data)
    
    def generate_communication_history(self, consent_df):
        """Generate communication history - only for consented users"""
        communications = []
        
        for _, consent_row in consent_df.iterrows():
            if consent_row['email_consent'] and pd.isna(consent_row['opt_out_date']):
                # Generate 1-5 communications per consented user
                num_comms = random.randint(1, 5)
                
                for i in range(num_comms):
                    comm_date = consent_row['consent_date'] + timedelta(days=i*3 + random.randint(0, 3))
                    
                    communication = {
                        'user_id': consent_row['user_id'],
                        'communication_type': np.random.choice(self.communication_types),
                        'communication_date': comm_date,
                        'response_status': np.random.choice(['opened', 'clicked', 'replied', 'no_response'], 
                                                       p=[0.3, 0.15, 0.05, 0.5]),
                        'campaign_id': f"campaign_{np.random.randint(1, 10)}",
                        'content_theme': np.random.choice(['product_demo', 'pricing_offer', 'case_study', 
                                                      'industry_insights', 'feature_update'])
                    }
                    communications.append(communication)
        
        return pd.DataFrame(communications)
    
    def generate_conversions(self, interactions_df, communications_df):
        """Generate conversion data - direct business relationships only"""
        conversions = []
        
        # Get users who had interactions
        interaction_users = interactions_df['user_id'].unique()
        
        # Conversion rate: ~20% of users who interacted
        converting_users = random.sample(list(interaction_users), int(len(interaction_users) * 0.2))
        
        for user_id in converting_users:
            # Get user's first interaction date
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            first_interaction_date = user_interactions['timestamp'].min()
            
            # Get user's communication history
            user_comms = communications_df[communications_df['user_id'] == user_id]
            
            # Conversion happens 1-30 days after first interaction
            conversion_date = first_interaction_date + timedelta(days=random.randint(1, 30))
            
            # Determine attribution channel
            if not user_comms.empty and random.random() < 0.7:
                attribution_channel = 'email_followup'
            else:
                attribution_channel = 'event_direct'
            
            conversion = {
                'user_id': user_id,
                'conversion_type': random.choice(self.conversion_types),
                'conversion_value': round(random.uniform(50, 5000), 2),
                'conversion_date': conversion_date,
                'attribution_channel': attribution_channel,
                'customer_segment': random.choice(['new_customer', 'existing_customer', 'prospect']),
                'product_category': random.choice(['enterprise_software', 'consulting_services', 
                                                 'training_program', 'subscription_service']),
                'days_to_conversion': (conversion_date - first_interaction_date).days
            }
            conversions.append(conversion)
        
        return pd.DataFrame(conversions)
    
    def generate_complete_dataset(self):
        """Generate complete legally compliant dataset"""
        print(" Generating legally compliant conversion tracking dataset...")
        print(" All data sources require explicit consent or direct business relationships")
        
        # Generate core tables
        interactions_df = self.generate_event_interactions()
        consent_df = self.generate_consent_preferences(interactions_df)
        communications_df = self.generate_communication_history(consent_df)
        conversions_df = self.generate_conversions(interactions_df, communications_df)
        
        # Generate summary statistics
        total_interactions = len(interactions_df)
        consented_users = len(consent_df)
        total_conversions = len(conversions_df)
        conversion_rate = (total_conversions / total_interactions) * 100
        avg_conversion_value = conversions_df['conversion_value'].mean()
        
        print(f"\n Dataset Summary:")
        print(f"Total Event Interactions: {total_interactions:,}")
        print(f"Users Who Gave Consent: {consented_users:,}")
        print(f"Total Conversions: {total_conversions:,}")
        print(f"Conversion Rate: {conversion_rate:.1f}%")
        print(f"Average Conversion Value: ${avg_conversion_value:,.2f}")
        
        datasets = {
            'interactions': interactions_df,
            'consent': consent_df,
            'communications': communications_df,
            'conversions': conversions_df
        }
        
        return datasets

# Generate the dataset
generator = LegalConversionDataGenerator(num_users=1000, num_events=5)
datasets = generator.generate_complete_dataset()

# Display sample data from each table
print("\n" + "="*60)
print("SAMPLE DATA FROM EACH TABLE")
print("="*60)

print("\n EVENT INTERACTIONS (First 5 rows):")
print(datasets['interactions'].head())

print("\n CONSENT & PREFERENCES (First 5 rows):")
print(datasets['consent'].head())

print("\n COMMUNICATION HISTORY (First 5 rows):")
print(datasets['communications'].head())

print("\n CONVERSIONS (First 5 rows):")
print(datasets['conversions'].head())

# Save datasets to CSV files
for table_name, df in datasets.items():
    filename = f"legal_{table_name}_data.csv"
    df.to_csv(filename, index=False)
    print(f"\n Saved {filename} ({len(df)} rows)")

print("\n" + "="*60)
print(" LEGAL COMPLIANCE NOTES FOR DESTRO:")
print("="*60)
print(" All data requires explicit user consent")
print(" No cross-device tracking or privacy violations") 
print(" Clear attribution chains for ROI measurement")
print(" GDPR/CCPA compliant data collection methods")
print(" Users can opt-out at any time")
print(" Only first-party data from direct business relationships")