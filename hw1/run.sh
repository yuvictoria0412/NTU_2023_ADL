
param1="$1"
param2="$2"
param3="$3"

# Print or perform operations with the parameters
# echo "The first parameter is: $param1"
# echo "The second parameter is: $param2"
# echo "The third parameter is: $param3"

# context, test, output, mc model, qa model

python "inference.py" "${1}" "${2}" "${3}" adl-hw1/output_roberta_epoch_3 adl-hw1/sample_output_mbl_n2
