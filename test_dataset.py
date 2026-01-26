#!/usr/bin/env python3
"""
Test script to validate MicrobiomeDataset implementation with existing data.
"""

import sys
import os

# Add src to path so we can import microfactual
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from microfactual import MicrobiomeDataset
    print("✅ Successfully imported MicrobiomeDataset")
except ImportError as e:
    print(f"❌ Failed to import MicrobiomeDataset: {e}")
    sys.exit(1)

def test_dataset_loading():
    """Test loading dataset from files."""
    print("\n🧪 Testing MicrobiomeDataset loading...")
    
    try:
        # Load dataset using your existing data
        dataset = MicrobiomeDataset.from_files(
            abundance_file="datasets/abundance_crc.txt",
            metadata_file="datasets/metadata_crc.txt", 
            target_column="Group",  # Assuming this is your target column
            sample_column="Sample ID"
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   {dataset}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None

def test_dataset_properties(dataset):
    """Test dataset properties and methods."""
    if dataset is None:
        return
        
    print("\n🧪 Testing dataset properties...")
    
    try:
        # Test basic properties
        print(f"✅ X shape (samples x features): {dataset.X.shape}")
        print(f"✅ y shape (target vector): {dataset.y.shape}")
        print(f"✅ Feature names: {len(dataset.feature_names)} features")
        print(f"✅ Sample names: {len(dataset.sample_names)} samples")
        print(f"✅ Target classes: {dataset.target_names}")
        
        # Test dataset info
        info = dataset.get_info()
        print(f"✅ Dataset info:")
        print(f"   - Samples: {info['n_samples']}")
        print(f"   - Features: {info['n_features']}")
        print(f"   - Sparsity: {info['sparsity']:.2%}")
        print(f"   - Classes: {info['target_classes']}")
        
    except Exception as e:
        print(f"❌ Error testing properties: {e}")

def test_preprocessing(dataset):
    """Test preprocessing methods."""
    if dataset is None:
        return
        
    print("\n🧪 Testing preprocessing methods...")
    
    try:
        # Test filtering (create a copy to not modify original)
        original_features = dataset.abundance.shape[0]
        filtered_dataset = dataset.filter_features(inplace=False)
        
        if filtered_dataset:
            filtered_features = filtered_dataset.abundance.shape[0]
            print(f"✅ Filtering: {original_features} → {filtered_features} features")
        
        # Test CLR transform  
        clr_dataset = dataset.clr_transform(inplace=False)
        if clr_dataset:
            print(f"✅ CLR transform: Applied successfully")
            print(f"   Preprocessing history: {len(clr_dataset._preprocessing_history)} steps")
        
    except Exception as e:
        print(f"❌ Error testing preprocessing: {e}")

def test_sklearn_compatibility(dataset):
    """Test sklearn compatibility."""
    if dataset is None:
        return
        
    print("\n🧪 Testing sklearn compatibility...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Get data in sklearn format
        X = dataset.X
        y = dataset.y
        
        print(f"✅ Data shapes compatible with sklearn:")
        print(f"   X: {X.shape}, y: {y.shape}")
        
        # Quick test with a small RandomForest
        if len(X) > 10:  # Only test if we have enough samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            rf = RandomForestClassifier(n_estimators=10, random_state=42)
            rf.fit(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Quick sklearn test passed:")
            print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"   Accuracy: {accuracy:.3f}")
        
    except Exception as e:
        print(f"❌ Error testing sklearn compatibility: {e}")

def main():
    """Run all tests."""
    print("🚀 Testing MicrobiomeDataset implementation...\n")
    
    # Test 1: Loading
    dataset = test_dataset_loading()
    
    # Test 2: Properties  
    test_dataset_properties(dataset)
    
    # Test 3: Preprocessing
    test_preprocessing(dataset)
    
    # Test 4: Sklearn compatibility
    test_sklearn_compatibility(dataset)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()