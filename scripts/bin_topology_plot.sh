ANALYSIS_ID=$(date -u +"%Y%m%dT%H%MZ")
THRESH=$1

SGB_DEF_FILE='/home/btheilma/code/NeuralTDA/standard_good_binnings.txt'
BIN_ID=$(date -u +"%Y%m%dT%H%MZ")
NSHUFFS=1
NCELLS=40
NPERMS=30

MAXBETTI=4
MAXT=7
FIGX=22
FIGY=22

echo "Topological Analysis with Threshold:" $1

for BLOCK_PATH in */ ; do

	echo "Binning Data..."
	bin_data $BLOCK_PATH $SGB_DEF_FILE $BIN_ID $NSHUFFS

	echo "Permuting Data..."
	permute_data_recursive $BLOCK_PATH/binned_data/$BIN_ID $NCELLS $NPERMS

	echo "Shuffling Permuted Data..."
	shuffle_data_recursive $BLOCK_PATH/binned_data/$BIN_ID/permuted_binned $NSHUFFS

	BIN_PATH=$BLOCK_PATH/binned_data/$BIN_ID
	PERMUTED_BIN_PATH=$BIN_PATH/permuted_binned
	SHUFFLED_PERMUTED_BIN_PATH=$PERMUTED_BIN_PATH/shuffled_controls

	echo "Computing Topology on Permuted Binned Data..."
	ANALYSIS_IDREAL=$ANALYSIS_ID_real
	calc_CI_topology_recursive $ANALYSIS_IDREAL $THRESH $BLOCK_PATH $PERMUTED_BIN_PATH

	echo "Computing Topology on Shuffled Permuted Binned Data"
	ANALYSIS_IDSHUFFLED=$ANALYSIS_ID_shuffled
	calc_CI_topology_recursive $ANALYSIS_IDSHUFFLED $THRESH $BLOCK_PATH $SHUFFLED_PERMUTED_BIN_PATH
	
	echo "Making Plots..."
	make_plots $BLOCK_PATH $ANALYSIS_ID $MAXBETTI $MAXT $FIGX $FIGY
done