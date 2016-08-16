ANALYSIS_ID=$(date +"%m%d%y%H%M")
BIN_ID=$1
THRESH=$2

for BLOCK_PATH in */ ; do

	BIN_PATH=$BLOCK_PATH/binned_data/$BIN_ID
	PERMUTED_BIN_PATH=$BIN_PATH/permuted_binned
	SHUFFLED_PERMUTED_BIN_PATH=$PERMUTED_BIN_PATH/shuffled_controls

	echo "Computing Topology on Permuted Binned Data..."
	ANALYSIS_ID_REAL=$ANALYSIS_ID_real
	calc_CI_topology_recursive $ANALYSIS_ID_REAL $THRESH $BLOCK_PATH $PERMUTED_BIN_PATH

	echo "Computing Topology on Shuffled Permuted Binned Data"
	ANALYSIS_ID_SHUFFLED=$ANALYSIS_ID_shuffled
	calc_CI_topology_recursive $ANALYSIS_ID_SHUFFLED $THRESH $BLOCK_PATH $SHUFFLED_PERMUTED_BIN_PATH
	
done
