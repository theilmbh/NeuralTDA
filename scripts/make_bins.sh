SGB_DEF_FILE='/home/btheilma/code/NeuralTDA/standard_good_binnings.txt'
BIN_ID=$(date +"%m%d%y%H%M")
NSHUFFS=1
NCELLS=40
NPERMS=30

for BLOCK_PATH in */ ; do
	echo "Binning Data..."
	bin_data $BLOCK_PATH $SGB_DEF_FILE $BIN_ID $NSHUFFS

	echo "Permuting Data..."
	permute_data_recursive $BLOCK_PATH/binned_data/$BIN_ID $NCELLS $NPERMS

	echo "Shuffling Permuted Data..."
	shuffle_data_recursive $BLOCK_PATH/binned_data/$BIN_ID/permuted_binned $NSHUFFS
done
