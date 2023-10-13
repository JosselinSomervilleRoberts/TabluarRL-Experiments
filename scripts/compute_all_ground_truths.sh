# For all foder names <folder_name> in ../data/
# Run python scripts/compute_ground_truth.py --env_name <folder_name>

for folder_name in $(ls data/); do
    echo "-----------------------------------"
    echo $folder_name
    PYTHONPATH=. python scripts/compute_ground_truth.py --env_name $folder_name
    echo "-----------------------------------\n"
done