import numpy as np

from logger import Logger

logger = Logger(log_level='DEBUG')

mri_contrasts = ["FLAIR", "T1w", "T1Gd", "T2w"]

def verify_mask_algo(miss_info, miss_prob_expected):
    # Verify proportion of masks is as expected
    miss_prob_actual = np.mean(miss_info, axis=0)
    logger.debug(f"Actual: {dict(zip(mri_contrasts, miss_prob_actual))}")
    logger.debug(f"Expected: {dict(zip(mri_contrasts, miss_prob_expected))}")
    mse_error = np.mean(np.square(miss_prob_actual-miss_prob_expected))
    logger.info(f"MSE: {mse_error}")

    # Verify that all-masking is not present

    count_all_true_rows = np.sum(np.all(miss_info, axis=1))
    if count_all_true_rows == 0:
        logger.debug("Excellent! No all-mask rows present!")
    else:
        logger.error(f"There are {count_all_true_rows} all-mask rows present!!")

# Masking Algo 1
class Mask1():
    def __init__(self, num_samples, miss_prob_expected, seed):
        np.random.seed(seed)
        self._num_samples = num_samples
        self._miss_prob_expected = miss_prob_expected
        self._num_contrasts = len(self._miss_prob_expected)
        self.miss_info = self._create()

    def _create(self):
        miss_info = np.ones((self._num_samples, self._num_contrasts), dtype=bool)
        all_true_rows = np.ones(self._num_samples, dtype=bool)
        while np.sum(all_true_rows) > 0:
            miss_info[all_true_rows] = np.random.rand(
                np.sum(all_true_rows), self._num_contrasts) < self._miss_prob_expected
            all_true_rows = np.all(miss_info, axis=1)
        return miss_info

    def verify(self):
        verify_mask_algo(self.miss_info, self._miss_prob_expected)

    def get_mask(self):
        return self.miss_info
    
# Masking Algo 0
class Mask0():
    def __init__(self, num_samples, num_contrasts, seed):
        np.random.seed(seed)
        self._num_samples = num_samples
        self._num_contrasts = num_contrasts
        nc = num_contrasts
        self._miss_prob_expected = np.array([(2**(nc-1)-1)/(2**nc-1)]*nc)
        self.miss_info = self._create()

        
    def _int_to_bool_binary(self, int_list, length):
        # Convert each integer to its base-2 value and represent it as boolean, always ensuring length is 4
        bool_list = []
        
        for num in int_list:
            # Get the binary representation of the integer (excluding the '0b' prefix)
            binary_str = bin(num)[2:]
            # Convert each character in the binary string to a boolean
            bools = [char == '1' for char in binary_str]
            # Prepend False (0s) to make the length exactly 4
            bools_padded = [False] * (length - len(bools)) + bools
            bool_list.append(bools_padded)
        return np.array(bool_list)
    
    def _create(self):
        mask_drop_code = np.random.randint(0, 2**(self._num_contrasts) - 1, size=self._num_samples)
        miss_info = self._int_to_bool_binary(mask_drop_code, length=self._num_contrasts)
        return miss_info
    
    def get(self, index):
        return self.miss_info[index]

    def verify(self):
        verify_mask_algo(self.miss_info, self._miss_prob_expected)

    def get_mask(self):
        return self.miss_info