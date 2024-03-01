# Default target
all: prepare_dataset apply_patch

prepare_dataset: decrypt_dataset extract_dataset clean

# Decrypt the dataset
decrypt_dataset:
	gpg --batch --passphrase benchmark -o data/dataset.tar -d data/dataset.tar.gpg

# Extract the dataset
extract_dataset: decrypt_dataset
	tar -xf data/dataset.tar -C data

# Clean up the tar file
clean: extract_dataset
	rm data/dataset.tar

# Apply the patch
apply_patch:
	bash accelerate_patch/apply_patch.sh

.PHONY: all prepare_dataset apply_patch