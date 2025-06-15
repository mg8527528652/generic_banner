#!/usr/bin/env python3
"""
Test script to demonstrate text overlap detection and fixing
"""

import json
import sys
from utils.composer_engine import validate_text_overlaps, fix_text_overlaps

def test_overlap_fix(banner_file):
    """Test overlap detection and fixing on a banner"""
    print(f"🧪 Testing overlap detection and fixing on {banner_file}")
    print("=" * 60)
    
    # Load banner data
    with open(banner_file, 'r') as f:
        data = json.load(f)
    
    resolution = [data.get('width', 1024), data.get('height', 1024)]
    
    # Test overlap detection
    print("📋 BEFORE FIX:")
    is_valid, errors = validate_text_overlaps(data)
    print(f"   Overlap Status: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    if errors:
        text_errors = [error for error in errors if 'textbox' in error]
        print(f"   Text Overlaps Found: {len(text_errors)}")
        for i, error in enumerate(text_errors[:3]):
            print(f"   {i+1}. {error[:100]}...")
        if len(text_errors) > 3:
            print(f"   ... and {len(text_errors) - 3} more text overlaps")
    
    # Apply fix
    print("\n🔧 APPLYING FIXES...")
    fixed_data = fix_text_overlaps(data.copy(), resolution)
    
    # Test after fix
    print("\n📋 AFTER FIX:")
    is_valid_after, errors_after = validate_text_overlaps(fixed_data)
    print(f"   Overlap Status: {'✅ PASS' if is_valid_after else '❌ FAIL'}")
    
    if errors_after:
        text_errors_after = [error for error in errors_after if 'textbox' in error]
        print(f"   Remaining Text Overlaps: {len(text_errors_after)}")
        for i, error in enumerate(text_errors_after[:3]):
            print(f"   {i+1}. {error[:100]}...")
    else:
        print("   ✅ All text overlaps resolved!")
    
    # Save fixed version
    fixed_file = banner_file.replace('.json', '_fixed.json')
    with open(fixed_file, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    print(f"\n💾 Fixed version saved to: {fixed_file}")
    
    return len(errors) - len(errors_after)

if __name__ == "__main__":
    # Test on problematic banners
    test_files = [
        'generated_banners/generated_banner_1.json',
        'generated_banners/generated_banner_2.json'
    ]
    
    total_fixes = 0
    for banner_file in test_files:
        try:
            fixes = test_overlap_fix(banner_file)
            total_fixes += fixes
            print()
        except Exception as e:
            print(f"❌ Error testing {banner_file}: {e}")
    
    print(f"🎯 Summary: Fixed {total_fixes} text overlap issues total") 