import sys
from utils.diarization_client import DiarizationClient

# Initialize diarization client
diarization_client = DiarizationClient("http://192.168.222.2:8000/diarize")

if __name__ == "__main__":
    # Ensure speaker name is provided as command-line argument
    name = ""
    if len(sys.argv) < 2:
        name = input("Please enter the name of the speaker to delete: ")
    else:
        name = sys.argv[1]
    
    if name == "":
        print("❌ No speaker name provided")
        sys.exit(1)

    # Confirm deletion
    confirm = input(f"Are you sure you want to delete speaker '{name}'? (y/n): ")
    if confirm.lower() not in ['y', 'yes']:
        print("Deletion cancelled")
        sys.exit(0)
    
    # Delete the speaker using DiarizationClient
    success = diarization_client.delete_speaker(name)
    
    if success:
        print(f"✅ Successfully deleted speaker: {name}")
    else:
        print(f"❌ Failed to delete speaker: {name}")
        sys.exit(1)
