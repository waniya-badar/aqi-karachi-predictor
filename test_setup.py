"""
Setup Testing Script
Run this to verify your setup is correct before starting the project
"""

import os
import sys
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def test_python_version():
    """Test Python version"""
    print("Testing Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 9:
        print(f" Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f" Python {version.major}.{version.minor}.{version.micro} - Need 3.9+")
        return False


def test_imports():
    """Test if required packages are installed"""
    print("\nTesting required packages...")
    
    packages = {
        'requests': 'requests',
        'pymongo': 'pymongo',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'dotenv': 'python-dotenv',
        'streamlit': 'streamlit',
        'plotly': 'plotly'
    }
    
    all_ok = True
    
    for module, package in packages.items():
        try:
            __import__(module)
            print(f" {package} - Installed")
        except ImportError:
            print(f" {package} - Missing")
            all_ok = False
    
    if not all_ok:
        print("\n💡 Install missing packages:")
        print("   pip install -r requirements.txt")
    
    return all_ok


def test_env_file():
    """Test if .env file exists and has required variables"""
    print("\nTesting environment variables...")
    
    if not os.path.exists('.env'):
        print(" .env file not found")
        print("\n Create .env file with:")
        print("   AQICN_API_KEY=your_key")
        print("   MONGODB_URI=your_uri")
        print("   MONGODB_DB_NAME=aqi_karachi")
        print("   KARACHI_STATION_ID=@8762")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        'AQICN_API_KEY',
        'MONGODB_URI',
        'MONGODB_DB_NAME',
        'KARACHI_STATION_ID'
    ]
    
    all_ok = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive data
            if 'KEY' in var or 'URI' in var:
                masked = value[:10] + '...' + value[-4:] if len(value) > 14 else '***'
                print(f"✅ {var} - Set ({masked})")
            else:
                print(f"✅ {var} - Set ({value})")
        else:
            print(f" {var} - Not set")
            all_ok = False
    
    return all_ok


def test_mongodb_connection():
    """Test MongoDB connection"""
    print("\nTesting MongoDB connection...")
    
    try:
        from src.mongodb_handler import MongoDBHandler
        
        handler = MongoDBHandler()
        stats = handler.get_data_statistics()
        handler.close()
        
        print(f" MongoDB connected successfully")
        print(f"   Database: {os.getenv('MONGODB_DB_NAME')}")
        print(f"   Records: {stats['total_records']}")
        return True
        
    except Exception as e:
        print(f" MongoDB connection failed: {e}")
        print("\n Check:")
        print("   1. MongoDB URI is correct in .env")
        print("   2. IP address is whitelisted in MongoDB Atlas")
        print("   3. Database user has correct permissions")
        return False


def test_api_connection():
    """Test AQICN API connection"""
    print("\nTesting AQICN API connection...")
    
    try:
        from src.data_fetcher import AQICNFetcher
        
        fetcher = AQICNFetcher()
        data = fetcher.fetch_current_data()
        
        if data and 'aqi' in data:
            print(f" AQICN API connected successfully")
            print(f"   Station: {data.get('station_name', 'Karachi')}")
            print(f"   Current AQI: {data['aqi']}")
            return True
        else:
            print(f" AQICN API returned no data")
            return False
            
    except Exception as e:
        print(f" AQICN API connection failed: {e}")
        print("\n Check:")
        print("   1. API key is correct in .env")
        print("   2. Station ID is correct")
        print("   3. Internet connection is working")
        return False


def test_directory_structure():
    """Test if all required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data/raw',
        'models/saved_models',
        'notebooks',
        'pipelines',
        'src',
        'streamlit_app',
        '.github/workflows'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f" {dir_path}/ - Exists")
        else:
            print(f" {dir_path}/ - Missing")
            all_ok = False
    
    return all_ok


def test_github_setup():
    """Test if Git is initialized"""
    print("\nTesting Git setup...")
    
    if os.path.exists('.git'):
        print(" Git repository initialized")
        
        # Try to get remote
        try:
            import subprocess
            result = subprocess.run(['git', 'remote', '-v'], 
                                  capture_output=True, text=True)
            if result.stdout:
                print(" GitHub remote configured")
                return True
            else:
                print("  No GitHub remote configured")
                print("\n Add remote:")
                print("   git remote add origin https://github.com/YOUR_USERNAME/aqi-predictor-karachi.git")
                return True
        except:
            return True
    else:
        print(" Git not initialized")
        print("\n Initialize Git:")
        print("   git init")
        return False


def run_feature_pipeline_test():
    """Test running feature pipeline"""
    print("\nTesting feature pipeline...")
    
    try:
        from pipelines.feature_pipeline import run_feature_pipeline
        
        print("Running feature pipeline (this may take 10-30 seconds)...")
        success = run_feature_pipeline()
        
        if success:
            print(" Feature pipeline test passed")
            return True
        else:
            print(" Feature pipeline test failed")
            return False
            
    except Exception as e:
        print(f" Feature pipeline error: {e}")
        return False


def main():
    """Run all tests"""
    print_header("AQI Predictor Setup Test")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'Python Version': test_python_version(),
        'Packages': test_imports(),
        'Environment Variables': test_env_file(),
        'Directory Structure': test_directory_structure(),
        'Git Setup': test_github_setup(),
    }
    
    # Only test connections if env is set up
    if results['Environment Variables']:
        results['MongoDB Connection'] = test_mongodb_connection()
        results['AQICN API'] = test_api_connection()
    
    # Only test pipeline if everything else works
    if all([results.get('MongoDB Connection'), results.get('AQICN API')]):
        print("\n" + "="*60)
        response = input("Run feature pipeline test? (collects real data) [y/N]: ")
        if response.lower() == 'y':
            results['Feature Pipeline'] = run_feature_pipeline_test()
    
    # Print summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = " PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed!")
    else:
        print("\n  Some tests failed. Fix the issues above.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()