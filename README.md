3/25/2020 - This project is on hold.  Doing a C# version instead (for now).
R Keene

# TWIT - Tensor Weighted Interpolative Transfer

(Release 1.0.  Updated the README.md)

(Update 12/2/2019 Multithread C code update.)
(This is changeing quite a bit due to the port to C so python can call multithreaded C functions for TWIT.  Stay tuned....  10/19/2019 RK)
(Updated 11/12/2019 - C code works well. 200x faster than pure python code.)

Richard Keene December 2, 2019
python PyPI package https://pypi.org/project/twit-rkeene/
Python 3 only.

# Concept of TWIT
The purpose of TWIT is to allow transfer of values from a source tensor to a destination tensor
with correct interpolation of values.  The source and destination do not have to have the same number of dimensions, 
e.g. len(src.shape) != len(dst.shape).
Also the range of indices do not have to match in count.  For example given two one dimensional tensors (vectors of values)
one could say copy from source range (2,7) to destination range (0,2) and use source to destination multipliers of (0.5, 0.9).  This will
copy source values at indices 2,3,4,5,6,7 to destination indices 0,1,2 (and 'scale' the data) and multiply 
source[2] by 0.5 to go to destination[0] and linear interpolate the multiplier (weight) up to 0.9 for subsequent indices.

### Energy Conservation
In the above example of scaling down the naive approach would just sum in the source values multiplied by the interpolate
weights would result in a destination that is 6/3 or 2x 'brighter' than the source.
What TWIT does is maintain constant energy or brightness while scaling values down by the source to destination ratio if the destination is shorter. If the destination is equal or longer then values are simply interpolated.  This maintains 'brightness' for both
up and down scale.

An intuitive example is a 3D tensor which happens to be a color image, going to a destination tensor 
that is a 2D grey scale image.  We want to maintain brightness and the dimensions do not match.

This image scale operation might be a source image that is 300x400x3 to a destination that is 400x600.  Nothing matches.
Lets take the example of input to TWIT of 
*Note: One can use python square brackets or parenthisis for clarity, twit does not care.*

# Typical twit call.
make_and_apply_twit(1, np.array([0, 4, 0, 3], dtype=np.int64),  np.array([1.0, 1.0], dtype=np.float64), t1, t2, 1)

This takes a tensor t1 that is 1D and uses 0 to 4 inclusive as the source, and t2 of 1D indicies 0 to 3 inclusive,
and copies the values from t1 to t2. The interpolation along the 1D dimension is 1.0 to 1.0, so no interpolation.
The final parameter 1 indicates preclear the destination to 0.0 before doing the transfer.

A more complex call might be
make_and_apply_twit(2, np.array([0, 4, 0, 3, 0, 2, 0, 4], dtype=np.int64),  np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float64), t1, t2, 1)

The first array is o to 4 in the source outer dimension, to 0 to 3 in the destination outer dimension, and 
0 to 2 in the source second innermost dimension, and 0 to 4 in the destination second dimension. Interpolation of the values ranges linearly 
from 0 to 1 along the first dimension, and 1 to 1 on the second dimension.

If the source and destination tensors are of different number of dimensions the lesser dimension tensor gets 1's filled in for the missing dimensions as a prepend to the shape of the smaller tensor.

# Reuse of twit results
If you are going to use the twit result repeatedly you should make the iterator, generate all the axis weights and indicies
once and then reuse the cached data. This is done with compute_twit_multi_dimension.  Then apply_twit can be called repeatedly.

# Why TWIT?
The motivation for TWIT is to support the development of the SRS cognition algorithm and Cognate.  The system has
lots of concept maps that are N dimensional tensors represent some concept space.  They are neurons and they are
connections between the concept maps.  Every system tick we do a twit transfer (using cached twit iterator values)
to transform data.  The fact this library can als be used for image scaling in tensors is a side effect.

# Python and C code
All of twit is written in python, and then the core is written again in C and multi-threaded.
twitc is the C interface to the DLL.
If you do not call twitc.set_thread_count then it will be single threaded C code.
You can call set_thread_count(N) where N is a power of 2, one of 2,4,8,16,32,64,128,256.
One can only call set_thread_count once.

# Speed issues.
twit in pure python will scale the test image shape (200,300,3) at about 3 seconds per apply_twit.
The single thread  code is about 1.8 milliseconds per apply_twit, and on an 8 core machine 
the multithread apply_twit is 0.5 milliseconds per apply_twit.

# Range strings - **Not reworked yet to new style of twit parameters!!!!!**
There is also a range definition string format used for Neural net concept map editing and viewing.
Input s is like "{5,17} <2,3> [-0.1, 0.9]" (see doc string for the module)
Format is ((srcstart, srcend), (dststart, dstend), (weightstart, weightend)).
src, dst, etc are to determine the ranges. The ..._shape_idx are which dimension of the array to use
for defaults.
Both indices and weights can be revered to indicate a reversed interpolation along that axis.
Indexes are inclusive "start value to end value", so {5, 9} means 5,6,7,8,9. 
This is in support of editors for people to easily enter ranges and weights for neural nets.
(See Cognate and NuTank)

# Helper Functions
TWIT includes some static functions to help do things.

make_and_apply_twit combines both making the twit cached data, and then applying it.

apply_twit(twt, t1, t2, preclear) will iterate the twit twt and do the copy and multiplies from t1 to t2.  If t1 and t2 are not the
same number of dimensions it will create the appropriate view and then do the work.  There is a clear destination flag to 
zero out the destination before iteration.  t2 MUST be a array style tensor since it gets written to in-place. You can pass
in an optional twit cache.

A little helper function if you need it, match_tensor_shape_lengths, will make the views and return them given a t1 and t2.

# Tests
There is a twit_test.py file with all the unit tests, and twitc_test.py.

# Examples
There is a scale_image_sample.py test file for an example.  It is not very generic. (Not yet done.)

# Source GitHub
At https://github.com/RMKeene/twit is the project.  Any improvements or bug fixes will be appreciated. Also available from PyPI.
