"""
Data Loader Module for Parkinson's Disease Predictor

Downloads and preprocesses UCI Parkinson's Voice Dataset and Oxford Telemonitoring Dataset.
Auto-downloads on first run with progress bar, caches to data/ folder.
"""

import os
import requests
import pandas as pd
import numpy as np
from io import StringIO
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """
    Handles downloading, caching, and preprocessing of Parkinson's datasets.
    
    Datasets:
    1. UCI Parkinson's Voice Dataset (195 samples, 23 features)
       - Voice measurements including jitter, shimmer, NHR, HNR, RPDE, DFA, PPE
       - Binary classification: 0 = healthy, 1 = Parkinson's
       
    2. Oxford Parkinson's Telemonitoring Dataset (5,875 samples, 22 features)
       - Longitudinal voice measurements with UPDRS scores
       - Regression target converted to binary for unified training
    """
    
    # Dataset URLs from UCI ML Repository
    UCI_VOICE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    OXFORD_TELEMONITORING_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
    
    # Unified feature set for voice analysis (common between datasets)
    VOICE_FEATURES = [
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
        'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]
    
    # Feature mapping for Oxford dataset (slightly different column names)
    OXFORD_FEATURE_MAP = {
        'Jitter(%)': 'Jitter(%)',
        'Jitter(Abs)': 'Jitter(Abs)',
        'Jitter:RAP': 'Jitter:RAP',
        'Jitter:PPQ5': 'Jitter:PPQ5',
        'Jitter:DDP': 'Jitter:DDP',
        'Shimmer': 'Shimmer',
        'Shimmer(dB)': 'Shimmer(dB)',
        'Shimmer:APQ3': 'Shimmer:APQ3',
        'Shimmer:APQ5': 'Shimmer:APQ5',
        'Shimmer:APQ11': 'Shimmer:APQ11',
        'Shimmer:DDA': 'Shimmer:DDA',
        'NHR': 'NHR',
        'HNR': 'HNR',
        'RPDE': 'RPDE',
        'DFA': 'DFA',
        'PPE': 'PPE'
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory to cache downloaded datasets
        """
        self.data_dir = data_dir
        self.uci_path = os.path.join(data_dir, "uci_parkinsons.csv")
        self.oxford_path = os.path.join(data_dir, "oxford_parkinsons.csv")
        self.combined_path = os.path.join(data_dir, "combined_parkinsons.csv")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def _download_file(self, url: str, description: str) -> str:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            description: Description for progress bar
            
        Returns:
            Downloaded content as string
        """
        print(f"\n📥 Downloading {description}...")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            content = []
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    content.append(chunk.decode('utf-8'))
                    pbar.update(len(chunk))
            
            return ''.join(content)
            
        except requests.RequestException as e:
            print(f"❌ Error downloading {description}: {e}")
            raise
    
    def download_uci_dataset(self, force: bool = False) -> pd.DataFrame:
        """
        Download UCI Parkinson's Voice Dataset.
        
        Args:
            force: If True, re-download even if cached
            
        Returns:
            DataFrame with UCI dataset
        """
        if os.path.exists(self.uci_path) and not force:
            print(f"✅ UCI dataset already cached at {self.uci_path}")
            return pd.read_csv(self.uci_path)
        
        content = self._download_file(self.UCI_VOICE_URL, "UCI Parkinson's Voice Dataset")
        
        # Parse CSV
        df = pd.read_csv(StringIO(content))
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # 'status' column: 1 = Parkinson's, 0 = healthy
        # Rename for consistency
        if 'status' in df.columns:
            df = df.rename(columns={'status': 'target'})
        
        # Save to cache
        df.to_csv(self.uci_path, index=False)
        print(f"✅ UCI dataset saved to {self.uci_path} ({len(df)} samples)")
        
        return df
    
    def download_oxford_dataset(self, force: bool = False) -> pd.DataFrame:
        """
        Download Oxford Parkinson's Telemonitoring Dataset.
        
        Args:
            force: If True, re-download even if cached
            
        Returns:
            DataFrame with Oxford dataset
        """
        if os.path.exists(self.oxford_path) and not force:
            print(f"✅ Oxford dataset already cached at {self.oxford_path}")
            return pd.read_csv(self.oxford_path)
        
        content = self._download_file(self.OXFORD_TELEMONITORING_URL, "Oxford Telemonitoring Dataset")
        
        # Parse CSV
        df = pd.read_csv(StringIO(content))
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Oxford dataset has motor_UPDRS and total_UPDRS as regression targets
        # Convert to binary classification: UPDRS > median = Parkinson's progression
        # For this dataset, all subjects have Parkinson's, so we use UPDRS for severity
        # We'll create binary target: 1 = high severity (needs attention), 0 = lower severity
        if 'motor_UPDRS' in df.columns:
            median_updrs = df['motor_UPDRS'].median()
            df['target'] = (df['motor_UPDRS'] > median_updrs).astype(int)
        
        # Save to cache
        df.to_csv(self.oxford_path, index=False)
        print(f"✅ Oxford dataset saved to {self.oxford_path} ({len(df)} samples)")
        
        return df
    
    def _extract_voice_features(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Extract common voice features from a dataset.
        
        Args:
            df: Source DataFrame
            source: 'uci' or 'oxford'
            
        Returns:
            DataFrame with standardized voice features
        """
        # Find matching columns (case-insensitive partial match)
        available_features = []
        feature_data = {}
        
        for target_feature in self.VOICE_FEATURES:
            # Try exact match first
            if target_feature in df.columns:
                feature_data[target_feature] = df[target_feature]
                available_features.append(target_feature)
            else:
                # Try case-insensitive match
                for col in df.columns:
                    if col.lower().replace(' ', '') == target_feature.lower().replace(' ', ''):
                        feature_data[target_feature] = df[col]
                        available_features.append(target_feature)
                        break
        
        result = pd.DataFrame(feature_data)
        result['target'] = df['target']
        result['source'] = source
        
        return result
    
    def get_combined_dataset(self, force_download: bool = False) -> pd.DataFrame:
        """
        Get combined dataset from UCI and Oxford sources.
        
        Args:
            force_download: If True, re-download all datasets
            
        Returns:
            Combined DataFrame with unified features
        """
        # Check for cached combined dataset
        if os.path.exists(self.combined_path) and not force_download:
            print(f"✅ Combined dataset already cached at {self.combined_path}")
            return pd.read_csv(self.combined_path)
        
        print("\n" + "="*60)
        print("🔄 Loading Parkinson's Disease Datasets")
        print("="*60)
        
        # Download both datasets
        uci_df = self.download_uci_dataset(force=force_download)
        oxford_df = self.download_oxford_dataset(force=force_download)
        
        # Extract common voice features
        print("\n📊 Extracting common voice features...")
        uci_features = self._extract_voice_features(uci_df, 'uci')
        oxford_features = self._extract_voice_features(oxford_df, 'oxford')
        
        # Combine datasets
        combined = pd.concat([uci_features, oxford_features], ignore_index=True)
        
        # Handle missing values
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].median())
        
        # Remove any remaining rows with NaN
        combined = combined.dropna()
        
        # Save combined dataset
        combined.to_csv(self.combined_path, index=False)
        
        print(f"\n✅ Combined dataset created: {len(combined)} total samples")
        print(f"   - UCI samples: {len(uci_features)}")
        print(f"   - Oxford samples: {len(oxford_features)}")
        print(f"   - Features: {len(self.VOICE_FEATURES)}")
        print(f"   - Saved to: {self.combined_path}")
        
        return combined
    
    def get_feature_names(self) -> list:
        """Get list of voice feature names."""
        return self.VOICE_FEATURES.copy()
    
    def get_train_test_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get train/test split of the combined dataset.
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        df = self.get_combined_dataset()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['target', 'source']]
        X = df[feature_cols]
        y = df['target']
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        df = self.get_combined_dataset()
        
        return {
            'total_samples': len(df),
            'uci_samples': len(df[df['source'] == 'uci']),
            'oxford_samples': len(df[df['source'] == 'oxford']),
            'num_features': len(self.VOICE_FEATURES),
            'feature_names': self.VOICE_FEATURES,
            'target_distribution': df['target'].value_counts().to_dict(),
            'class_balance': df['target'].mean()
        }


# Convenience function for quick access
def load_parkinsons_data(data_dir: str = "data", force_download: bool = False) -> pd.DataFrame:
    """
    Convenience function to load combined Parkinson's dataset.
    
    Args:
        data_dir: Directory for caching datasets
        force_download: If True, re-download all datasets
        
    Returns:
        Combined DataFrame with voice features
    """
    loader = DataLoader(data_dir=data_dir)
    return loader.get_combined_dataset(force_download=force_download)


if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader...")
    loader = DataLoader()
    df = loader.get_combined_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nDataset info:")
    info = loader.get_dataset_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
