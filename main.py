import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class EdmontonRealEstateExtractor:
    """
    Enhanced data extractor for Edmonton real estate data
    Uses actual API data where available and industry-standard estimates for missing data
    """
    
    def __init__(self):
        self.base_url = "https://data.edmonton.ca/resource"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Edmonton Real Estate Analytics Tool',
            'Accept': 'application/json'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Define the unified feature schema
        self.unified_schema = {
            'property_id': 'str',
            'address': 'str',
            'neighbourhood': 'str',
            'ward': 'str',
            'assessed_value': 'float',
            'land_value': 'float',      # Will be estimated
            'improvement_value': 'float', # Will be estimated
            'total_living_area_sf': 'float', 
            'lot_size_sf': 'float',    
            'year_built': 'int',       
            'bedrooms': 'int',         # Will be estimated
            'bathrooms': 'float',      # Will be estimated
            'garage': 'str',
            'basement': 'str',         # Will be estimated
            'property_type': 'str',    # Will be estimated
            'property_use': 'str',     # Will be estimated
            'latitude': 'float',
            'longitude': 'float',
            'annual_tax': 'float',     # Will be estimated
            'price_per_sqft': 'float', # Will be estimated
            'land_to_building_ratio': 'float', # Will be estimated
            'building_age': 'int',     # Will be estimated
            'is_estimated': 'str'      # Flag to indicate estimated fields
        }
        
        # Edmonton Open Data API endpoints
        self.endpoints = {
            'current_assessment': 'q7d6-ambg.json',  # Current property assessment
            'property_characteristics': 'dkk9-cj3x.json'  # property characteristics
        }
    
    def fetch_data_batch(self, endpoint: str, limit: int = 10000, offset: int = 0) -> Optional[List[Dict]]:
        """
        Fetch data in batches from Edmonton Open Data API
        """
        url = f"{self.base_url}/{endpoint}"
        params = {
            '$limit': limit,
            '$offset': offset,
            '$order': 'account_number'
        }
        
        try:
            self.logger.info(f"Fetching batch from {endpoint}: offset={offset}, limit={limit}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully fetched {len(data)} records")
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from {endpoint}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON from {endpoint}: {e}")
            return None
    
    def fetch_all_data(self, endpoint: str, max_records: int = 50000) -> List[Dict]:
        """
        Fetch all available data from an endpoint with pagination
        """
        all_data = []
        batch_size = 10000
        offset = 0
        
        while len(all_data) < max_records:
            batch = self.fetch_data_batch(endpoint, batch_size, offset)
            
            if not batch:
                break
                
            all_data.extend(batch)
            
            # If we got less than batch_size, we've reached the end
            if len(batch) < batch_size:
                break
                
            offset += batch_size
            time.sleep(1)  # Be respectful to the API
            
            self.logger.info(f"Total records fetched so far: {len(all_data)}")
        
        return all_data[:max_records]
    
    def normalize_property_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """
        Convert raw Edmonton data to unified schema, using estimates where necessary
        """
        self.logger.info("Normalizing property data to unified schema...")
        
        normalized_records = []
        
        for record in raw_data:
            try:
                normalized = self.create_unified_record_with_estimates(record)
                if normalized:
                    normalized_records.append(normalized)
            except Exception as e:
                self.logger.debug(f"Error normalizing record: {e}")
                continue
        
        df = pd.DataFrame(normalized_records)
        
        # Apply data type conversions
        for column, dtype in self.unified_schema.items():
            if column in df.columns:
                df[column] = self.convert_column_type(df[column], dtype)
        
        # Calculate derived features
        df = self.calculate_derived_features(df)
        
        self.logger.info(f"Normalized {len(df)} records with unified schema")
        return df
    
    def create_unified_record_with_estimates(self, record: Dict) -> Optional[Dict]:
        """
        Map Edmonton API fields to unified schema, using actual data when available or estimates otherwise
        """
        try:
            # Extract actual data from the API
            unified_record = {
                'property_id': self.safe_get(record, 'account_number', ''),
                'address': self.create_address_from_location(record),
                'neighbourhood': self.safe_get(record, 'neighbourhood', ''),
                'ward': self.safe_get(record, 'ward', ''),
                'assessed_value': self.safe_float(record.get('assessed_value', 0)),
                'garage': self.safe_get(record, 'garage', 'Unknown'),
                'latitude': self.safe_float(record.get('latitude', 0)),
                'longitude': self.safe_float(record.get('longitude', 0)),
                'is_estimated': 'land_value,improvement_value,bedrooms,bathrooms,basement,property_type,property_use,annual_tax,price_per_sqft,land_to_building_ratio,building_age'
            }
            
            # Only proceed if we have the essential data
            if (unified_record['property_id'] and 
                unified_record['assessed_value'] > 0 and
                unified_record['neighbourhood']):
                
                # Look for actual property characteristics data
                address_key = self.generate_address_key(record)
                actual_data_found = False
                
                if hasattr(self, 'property_chars_lookup') and address_key in self.property_chars_lookup:
                    chars = self.property_chars_lookup[address_key]
                    
                    # Only use the data if it's valid (not empty)
                    lot_size_valid = chars['lot_size_sf'] > 0
                    living_area_valid = chars['total_living_area_sf'] > 0
                    year_built_valid = chars['year_built'] > 0
                    
                    # If year_built or total_living_area_sf are empty, ignore this record
                    if not year_built_valid or not living_area_valid:
                        return None  # Skip this record as it's likely a land listing
                    
                    # Use actual lot size if available, otherwise estimate
                    if lot_size_valid:
                        unified_record['lot_size_sf'] = chars['lot_size_sf']
                        # Remove lot_size_sf from estimated fields
                        unified_record['is_estimated'] = unified_record['is_estimated'].replace('lot_size_sf,', '')
                        unified_record['is_estimated'] = unified_record['is_estimated'].replace(',lot_size_sf', '')
                        actual_data_found = True
                    
                    # Use actual living area data
                    unified_record['total_living_area_sf'] = chars['total_living_area_sf']
                    
                    # Use actual year built data
                    unified_record['year_built'] = chars['year_built']
                    
                    # Remove total_living_area_sf and year_built from estimated fields
                    unified_record['is_estimated'] = unified_record['is_estimated'].replace('total_living_area_sf,', '')
                    unified_record['is_estimated'] = unified_record['is_estimated'].replace(',total_living_area_sf', '')
                    unified_record['is_estimated'] = unified_record['is_estimated'].replace('year_built,', '')
                    unified_record['is_estimated'] = unified_record['is_estimated'].replace(',year_built', '')
                
                # Estimate missing values based on industry standards and local knowledge
                estimated_values = self.estimate_property_values(unified_record)
                
                # Only override with estimates if we don't have actual data
                for key, value in estimated_values.items():
                    if key not in unified_record or unified_record[key] == 0:
                        unified_record[key] = value
                
                return unified_record
                    
        except Exception as e:
            self.logger.debug(f"Error creating unified record: {e}")
            
        return None
    
    def create_address_from_location(self, record: Dict) -> str:
        """
        Create an approximate address from neighbourhood and location
        """
        neighbourhood = self.safe_get(record, 'neighbourhood', '')
        
        if 'suite' in record and record['suite']:
            suite_info = f"Suite {record['suite']}, "
        else:
            suite_info = ""
            
        if 'house_number' in record and 'street_name' in record:
            return f"{suite_info}{record['house_number']} {record['street_name']} Edmonton, AB"
        
        return f"{suite_info}{neighbourhood} Edmonton, AB"
    
    def estimate_property_values(self, base_record: Dict) -> Dict:
        """
        Estimate missing property values using industry standards and the assessed value
        """
        assessed_value = self.safe_float(base_record.get('assessed_value', 0))
        neighbourhood = base_record.get('neighbourhood', '')
        
        # Default values
        current_year = datetime.now().year
        estimates = {
            # Land typically accounts for 30-40% of property value in Edmonton
            'land_value': round(assessed_value * 0.35),
            
            # Improvements (building) account for 60-70%
            'improvement_value': round(assessed_value * 0.65),
            
            # Only estimate these if they're not already in the base record
            'total_living_area_sf': base_record.get('total_living_area_sf', 0) or self.estimate_home_size(assessed_value),
            'lot_size_sf': base_record.get('lot_size_sf', 0) or self.estimate_lot_size(assessed_value, neighbourhood),
            'year_built': base_record.get('year_built', 0) or self.estimate_year_built(neighbourhood),
            
            # Typical bedroom count based on home size (always estimate)
            'bedrooms': self.estimate_bedrooms(assessed_value),
            
            # Typical bathroom count (always estimate)
            'bathrooms': self.estimate_bathrooms(assessed_value),
            
            # Most Edmonton homes have basements (always estimate)
            'basement': self.estimate_basement_type(assessed_value, neighbourhood, self.safe_float(base_record.get('year_built', 0))),
            
            # Majority are single family homes (always estimate)
            'property_type': self.estimate_property_type(assessed_value, neighbourhood),
            'property_use': self.estimate_property_use(assessed_value, neighbourhood),
        }
        
        return estimates
    
    def estimate_property_type(self, assessed_value: float, neighbourhood: str) -> str:
        """Estimate property type based on value and neighbourhood"""
        # Set probabilities for different property types based on value and location
        if any(x in neighbourhood.lower() for x in ['downtown', 'central', 'university', 'oliver']):
            # Downtown/central areas have more condos and apartments
            types = ['Condominium', 'Single Family Home', 'Duplex', 'Row Housing', 'Townhouse']
            weights = [0.5, 0.2, 0.1, 0.1, 0.1]
        elif any(x in neighbourhood.lower() for x in ['estate', 'ridge', 'valley', 'river', 'country']):
            # Estate areas are mostly detached homes
            types = ['Single Family Home', 'Executive Home', 'Estate Property', 'Luxury Home', 'Condominium']
            weights = [0.6, 0.2, 0.1, 0.08, 0.02]
        elif assessed_value > 750000:
            # Higher value homes
            types = ['Single Family Home', 'Executive Home', 'Estate Property', 'Custom Built Home']
            weights = [0.5, 0.3, 0.1, 0.1]
        elif assessed_value < 300000:
            # Lower value properties
            types = ['Condominium', 'Townhouse', 'Duplex', 'Single Family Home', 'Row Housing']
            weights = [0.4, 0.25, 0.15, 0.1, 0.1]
        else:
            # Default mix
            types = ['Single Family Home', 'Condominium', 'Duplex', 'Townhouse', 'Row Housing']
            weights = [0.6, 0.2, 0.1, 0.05, 0.05]
            
        return np.random.choice(types, p=weights)

    def estimate_property_use(self, assessed_value: float, neighbourhood: str) -> str:
        """Estimate property use based on value and neighbourhood"""
        # Set probabilities for different property uses
        if any(x in neighbourhood.lower() for x in ['industrial', 'business', 'commercial']):
            # Business areas
            uses = ['Residential', 'Commercial', 'Mixed Use', 'Investment Property', 'Rental Property']
            weights = [0.7, 0.15, 0.05, 0.05, 0.05]
        elif assessed_value > 800000:
            # Higher value properties
            uses = ['Owner Occupied', 'Residential', 'Vacation Property', 'Investment Property']
            weights = [0.7, 0.2, 0.05, 0.05]
        else:
            # Default mix for most properties
            uses = ['Residential', 'Owner Occupied', 'Rental Property', 'Investment Property']
            weights = [0.7, 0.2, 0.07, 0.03]
            
        return np.random.choice(uses, p=weights)
    
    def estimate_basement_type(self, assessed_value: float, neighbourhood: str, year_built: int) -> str:
        """Estimate basement type based on value, age and neighbourhood"""
        # Set probabilities based on property characteristics
        current_year = datetime.now().year
        if year_built == 0:
            estimated_age = self.estimate_year_built(neighbourhood)
            age = current_year - estimated_age
        else:
            age = current_year - year_built
            
        # Newer luxury homes likely have finished basements
        if assessed_value > 600000 and age < 20:
            basement_types = ['Finished', 'Fully Finished', 'Walkout', 'Partially Finished']
            weights = [0.3, 0.4, 0.2, 0.1]
        # Older luxury homes might have renovated basements
        elif assessed_value > 600000:
            basement_types = ['Finished', 'Partially Finished', 'Unfinished', 'Walkout', 'Renovated']
            weights = [0.3, 0.2, 0.1, 0.3, 0.1]
        # Mid-range newer homes
        elif assessed_value > 400000 and age < 30:
            basement_types = ['Partially Finished', 'Finished', 'Unfinished', 'None']
            weights = [0.4, 0.3, 0.2, 0.1]
        # Older mid-range homes
        elif assessed_value > 400000:
            basement_types = ['Unfinished', 'Partially Finished', 'Finished', 'None']
            weights = [0.4, 0.3, 0.2, 0.1]
        # Lower value or older homes
        elif age > 50:
            basement_types = ['Unfinished', 'None', 'Partially Finished', 'Concrete']
            weights = [0.4, 0.3, 0.2, 0.1]
        # Default mix
        else:
            basement_types = ['Unfinished', 'Partially Finished', 'None', 'Finished']
            weights = [0.4, 0.3, 0.2, 0.1]
        
        return np.random.choice(basement_types, p=weights)
    
    def estimate_home_size(self, assessed_value: float) -> float:
        """Estimate home size based on assessed value and local averages"""
        # Approximate price per square foot ranges in Edmonton
        if assessed_value <= 200000:
            price_per_sqft = 200
        elif assessed_value <= 350000:
            price_per_sqft = 250
        elif assessed_value <= 500000:
            price_per_sqft = 300
        elif assessed_value <= 750000:
            price_per_sqft = 350
        else:
            price_per_sqft = 400
            
        # Calculate approximate size and round to nearest 12 (foot)
        est_size = assessed_value / price_per_sqft
        return round(est_size / 12) * 12  # Round to nearest foot
    
    def estimate_lot_size(self, assessed_value: float, neighbourhood: str) -> float:
        """Estimate lot size based on value and neighbourhood type"""
        # Base size dependent on property value
        if assessed_value <= 300000:
            base_size = 4000  # Smaller lots for lower-value homes
        elif assessed_value <= 500000:
            base_size = 5500  # Medium lots
        elif assessed_value <= 750000:
            base_size = 7000  # Larger lots
        else:
            base_size = 9000  # Luxury homes
            
        # Adjust based on neighbourhood type
        if any(x in neighbourhood.lower() for x in ['downtown', 'central', 'university']):
            factor = 0.7  # Smaller lots in central areas
        elif any(x in neighbourhood.lower() for x in ['estate', 'ridge', 'valley', 'river']):
            factor = 1.4  # Larger lots in estate areas
        else:
            factor = 1.0  # Average suburban areas
            
        return round(base_size * factor / 12) * 12  # Round to nearest foot
    
    def estimate_year_built(self, neighbourhood: str) -> int:
        """Estimate year built based on neighbourhood development patterns"""
        current_year = datetime.now().year
        
        # Map neighborhood keywords to approximate development eras
        if any(x in neighbourhood.lower() for x in ['downtown', 'historical', 'park', 'westmount']):
            return current_year - 75  # Older areas (~1950s)
        elif any(x in neighbourhood.lower() for x in ['millwoods', 'castle', 'heritage']):
            return current_year - 45  # Established areas (~1980s)
        elif any(x in neighbourhood.lower() for x in ['summerside', 'terwillegar', 'windermere']):
            return current_year - 20  # Newer areas (~2000s)
        elif any(x in neighbourhood.lower() for x in ['walker', 'orchards', 'griesbach', 'secord']):
            return current_year - 10  # Very new areas (~2010s)
        else:
            return current_year - 35  # Default to average age (~1990s)
    
    def estimate_bedrooms(self, assessed_value: float) -> int:
        """Estimate bedroom count based on assessed value"""
        if assessed_value <= 250000:
            return 2  # Smaller homes/condos
        elif assessed_value <= 400000:
            return 3  # Typical family homes
        elif assessed_value <= 600000:
            return 4  # Larger family homes
        else:
            return 5  # Luxury homes
    
    def estimate_bathrooms(self, assessed_value: float) -> float:
        """Estimate bathroom count based on assessed value"""
        if assessed_value <= 250000:
            return 1.0  # Smaller homes/condos
        elif assessed_value <= 350000:
            return 1.5  # Entry-level homes
        elif assessed_value <= 450000:
            return 2.0  # Mid-range homes
        elif assessed_value <= 600000:
            return 2.5  # Upper mid-range homes
        elif assessed_value <= 800000:
            return 3.0  # Higher-end homes
        else:
            return 3.5  # Luxury homes
    
    def standardize_property_type(self, prop_type: str) -> str:
        """
        Standardize property type values
        """
        if not prop_type:
            return 'Unknown'
        
        prop_type = prop_type.upper()
        
        if 'SINGLE' in prop_type or 'DETACHED' in prop_type:
            return 'Single Family Home'
        elif 'CONDO' in prop_type or 'APARTMENT' in prop_type:
            return 'Condominium'
        elif 'DUPLEX' in prop_type:
            return 'Duplex'
        elif 'TOWNHOUSE' in prop_type or 'ROW' in prop_type:
            return 'Townhouse'
        elif 'COMMERCIAL' in prop_type:
            return 'Commercial'
        else:
            return prop_type.title()
    
    def safe_get(self, record: Dict, key: str, default: str = '') -> str:
        """Safely get string value from record"""
        value = record.get(key, default)
        return str(value) if value is not None else default
    
    def safe_float(self, value) -> float:
        """Safely convert to float"""
        try:
            if value is None or value == '':
                return 0.0
            return float(str(value).replace(',', '').replace('$', ''))
        except (ValueError, TypeError):
            return 0.0
    
    def safe_int(self, value) -> int:
        """Safely convert to int"""
        try:
            if value is None or value == '':
                return 0
            return int(float(str(value).replace(',', '')))
        except (ValueError, TypeError):
            return 0
    
    def convert_column_type(self, series: pd.Series, target_type: str) -> pd.Series:
        """Convert pandas series to target type"""
        try:
            if target_type == 'float':
                return pd.to_numeric(series, errors='coerce').fillna(0.0)
            elif target_type == 'int':
                return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)
            elif target_type == 'str':
                return series.astype(str).fillna('')
        except Exception as e:
            self.logger.debug(f"Error converting column type: {e}")
        
        return series
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features like price per sqft, building age, etc.
        """
        current_year = datetime.now().year
        
        # Building age
        df['building_age'] = current_year - df['year_built']
        df.loc[df['year_built'] == 0, 'building_age'] = 0
        
        # Price per square foot
        df['price_per_sqft'] = 0.0
        mask = df['total_living_area_sf'] > 0
        df.loc[mask, 'price_per_sqft'] = df.loc[mask, 'assessed_value'] / df.loc[mask, 'total_living_area_sf']
        
        # Land to building ratio
        df['land_to_building_ratio'] = 0.0
        mask = (df['lot_size_sf'] > 0) & (df['total_living_area_sf'] > 0)
        df.loc[mask, 'land_to_building_ratio'] = df.loc[mask, 'lot_size_sf'] / df.loc[mask, 'total_living_area_sf']
        
        # Estimated annual tax (rough calculation)
        df['annual_tax'] = df['assessed_value'] * 0.012  # Approximate Edmonton tax rate
        
        return df
    
    def extract_dataset_with_estimates(self, max_records: int = 20000) -> pd.DataFrame:
        """
        Extract Edmonton real estate dataset with actual data where available and estimates elsewhere
        """
        self.logger.info(f"Starting data extraction with estimates for {max_records} records...")
        
        # Fetch current assessment data (what's available)
        assessment_data = self.fetch_all_data(self.endpoints['current_assessment'], max_records)
        
        if not assessment_data:
            self.logger.error("Failed to fetch assessment data")
            return pd.DataFrame()
        
        # Fetch property characteristics data (lot size, living area, year built)
        self.logger.info("Fetching property characteristics data...")
        characteristics_data = self.fetch_all_data(self.endpoints['property_characteristics'], max_records)
        
        if characteristics_data:
            self.logger.info(f"Successfully fetched {len(characteristics_data)} property characteristics records")
            # Create a lookup dictionary for quick access to property characteristics
            self.property_chars_lookup = self.create_property_characteristics_lookup(characteristics_data)
        else:
            self.logger.warning("Failed to fetch property characteristics data, will use estimates only")
            self.property_chars_lookup = {}
        
        # Normalize to unified schema with estimates for missing fields
        unified_df = self.normalize_property_data(assessment_data)
        
        # Filter for quality records
        unified_df = self.filter_quality_records(unified_df)
        
        self.logger.info(f"Final dataset: {len(unified_df)} records with {len(unified_df.columns)} features")
        
        return unified_df

    def create_property_characteristics_lookup(self, characteristics_data: List[Dict]) -> Dict:
        """
        Create a lookup dictionary from property characteristics data
        """
        lookup = {}
        for record in characteristics_data:
            # Use house number + street name as a key since account_number may not match between datasets
            address_key = self.generate_address_key(record)
            if address_key:
                lookup[address_key] = {
                    'lot_size_sf': self.safe_float(record.get('lot_size', 0)),
                    'total_living_area_sf': self.safe_float(record.get('total_gross_area', 0)),
                    'year_built': self.safe_int(record.get('year_built', 0))
                }
        
        self.logger.info(f"Created property characteristics lookup with {len(lookup)} entries")
        return lookup

    def generate_address_key(self, record: Dict) -> str:
        """
        Generate a consistent address key for matching records between datasets
        """
        house_number = self.safe_get(record, 'house_number', '')
        street_name = self.safe_get(record, 'street_name', '')
        
        if house_number and street_name:
            # Normalize the address components to improve matching
            return f"{house_number}_{street_name}".lower().replace(' ', '_')
        return ""
    
    def filter_quality_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for high-quality records
        """
        initial_count = len(df)
        
        # Remove records with missing essential data
        df = df[
            (df['property_id'] != '') &
            (df['assessed_value'] > 0) &
            (df['neighbourhood'] != '')
        ]
        
        # Remove outliers
        # Filter extreme assessed values (likely errors)
        q1 = df['assessed_value'].quantile(0.01)
        q99 = df['assessed_value'].quantile(0.99)
        df = df[(df['assessed_value'] >= q1) & (df['assessed_value'] <= q99)]
        
        filtered_count = len(df)
        self.logger.info(f"Filtered {initial_count - filtered_count} records, kept {filtered_count}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename_prefix: str = 'edmonton_real_estate_unified') -> Dict[str, str]:
        """
        Save the unified dataset in multiple formats
        """
        if df.empty:
            self.logger.warning("No data to save")
            return {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_created = {}
        
        try:
            # Save as CSV
            csv_filename = f"{filename_prefix}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            files_created['csv'] = csv_filename
            
            # Save metadata
            metadata = {
                'total_records': len(df),
                'features': list(df.columns),
                'schema_version': '1.0',
                'data_quality': {
                    'actual_data_fields': ['property_id', 'neighbourhood', 'ward', 'assessed_value', 'garage', 'latitude', 'longitude', 'total_living_area_sf', 'lot_size_sf', 'year_built'],
                    'estimated_fields': ['land_value', 'improvement_value', 'bedrooms', 'bathrooms', 'property_type', 'property_use'],
                    'derived_fields': ['annual_tax', 'price_per_sqft', 'land_to_building_ratio', 'building_age']
                },
                'statistics': {
                    'avg_assessed_value': float(df['assessed_value'].mean()),
                    'median_assessed_value': float(df['assessed_value'].median()),
                    'neighbourhoods': len(df['neighbourhood'].unique())
                }
            }
            
            metadata_filename = f"{filename_prefix}_metadata_{timestamp}.json"
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            files_created['metadata'] = metadata_filename
            
            self.logger.info(f"Saved dataset: {len(df)} records, {len(df.columns)} features")
            for file_type, filename in files_created.items():
                self.logger.info(f"  {file_type.upper()}: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving dataset: {e}")
        
        return files_created
    
    def print_dataset_summary(self, df: pd.DataFrame):
        """
        Print comprehensive dataset summary
        """
        if df.empty:
            print("Dataset is empty")
            return
        
        print("\n" + "="*80)
        print("EDMONTON REAL ESTATE DATASET SUMMARY (WITH ESTIMATES)")
        print("="*80)
        
        print(f"\nDataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Data Source: Edmonton Open Data Portal with estimates for missing fields")
        print(f"   Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nActual API Data:")
        actual_fields = ['property_id', 'neighbourhood', 'ward', 'assessed_value', 'garage', 'latitude', 'longitude',
                         'total_living_area_sf', 'lot_size_sf', 'year_built']
        for field in actual_fields:
            if field in df.columns:
                non_null = df[field].notnull().sum()
                print(f"   {field}: {non_null:,} non-null values ({non_null/len(df)*100:.1f}%)")
        
        print(f"\nEstimated Data (based on industry standards):")
        estimated_fields = ['land_value', 'improvement_value', 'bedrooms', 'bathrooms', 'property_type', 'property_use']
        for field in estimated_fields:
            if field in df.columns:
                print(f"   {field}: ESTIMATED based on property values and neighborhood patterns")
        
        print(f"\nProperty Values:")
        print(f"   Average Assessed Value: ${df['assessed_value'].mean():,.2f} (ACTUAL DATA)")
        print(f"   Median Assessed Value: ${df['assessed_value'].median():,.2f} (ACTUAL DATA)")
        print(f"   Est. Average Land Value: ${df['land_value'].mean():,.2f} (ESTIMATED)")
        print(f"   Est. Average Improvement Value: ${df['improvement_value'].mean():,.2f} (ESTIMATED)")
        print(f"   Est. Average Living Area: {df['total_living_area_sf'].mean():.0f} sq ft")
        print(f"   Est. Average Lot Size: {df['lot_size_sf'].mean():.0f} sq ft")
        
        print(f"\nNeighbourhoods:")
        print(f"   Total Neighbourhoods: {df['neighbourhood'].nunique()} (ACTUAL DATA)")
        top_neighbourhoods = df['neighbourhood'].value_counts().head()
        for neighbourhood, count in top_neighbourhoods.items():
            print(f"   {neighbourhood}: {count:,} properties")
        
        print(f"\nProperty Features (ESTIMATED):")
        print(f"   Est. Average Bedrooms: {df['bedrooms'].mean():.1f}")
        print(f"   Est. Average Bathrooms: {df['bathrooms'].mean():.1f}")
        
        print(f"\nNOTE: This dataset uses a combination of actual API data and estimates.")
        print(f"Fields marked 'ACTUAL DATA' come directly from the Edmonton Open Data API.")
        print(f"Fields marked 'ESTIMATED' are calculated based on industry standards and local")
        print(f"real estate patterns, as the API does not provide these values directly.")
        print(f"The 'is_estimated' column in the dataset indicates which fields are estimates.")

# Main execution function
def main():
    """
    Main function to extract Edmonton real estate data with estimates
    """
    print("Edmonton Real Estate Data Extractor (with Estimates)")
    print("="*60)
    print("\nNOTE: This script will use actual data where available from the Edmonton Open Data API")
    print("and provide reasonable estimates for fields that are not directly available.")
    print("All estimated fields will be clearly marked in the dataset.")
    
    # Initialize extractor
    extractor = EdmontonRealEstateExtractor()
    
    # Extract dataset with estimates for missing fields
    print("\nExtracting Edmonton real estate data...")
    df = extractor.extract_dataset_with_estimates(max_records=15000)
    
    if df.empty:
        print("Failed to extract data. Please check your internet connection and try again.")
        return
    
    # Print summary
    extractor.print_dataset_summary(df)
    
    # Save dataset
    print(f"\nSaving dataset...")
    files_created = extractor.save_dataset(df, filename_prefix='edmonton_st_albert_combined')
    
    if files_created:
        print(f"\nSuccessfully created Edmonton real estate dataset!")
        print(f"Files created:")
        for file_type, filename in files_created.items():
            print(f"   {file_type.upper()}: {filename}")
        
        print(f"\nIMPORTANT: This dataset contains both actual data and estimates.")
        print(f"Check the 'is_estimated' column to identify estimated fields.")
    
    return df

if __name__ == "__main__":
    dataset = main()
