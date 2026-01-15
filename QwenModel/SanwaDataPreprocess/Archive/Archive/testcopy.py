import shutil
import os

# 1. Define one specific file from your error log
src_file = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/2025-12-16 18.23.26/ROI_16.jpg'

# 2. Define where it should go
dst_folder = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/manual_validate/test_folder/'
dst_file = os.path.join(dst_folder, 'ROI_16.jpg')

try:
    # Ensure the test folder exists
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"Created folder: {dst_folder}")

    # Try copying with shutil.copy (copies content + permission, ignores timestamp)
    shutil.copy(src_file, dst_file)
    
    print("------------------------------------------------")
    print("SUCCESS: File copied.")
    print(f"Source: {src_file}")
    print(f"Dest:   {dst_file}")
    print("------------------------------------------------")

except FileNotFoundError:
    print("ERROR: Source file not found. Check the path.")
except PermissionError:
    print("ERROR: Permission denied. You might not have write access to the destination drive.")
except Exception as e:
    print(f"ERROR: {e}")