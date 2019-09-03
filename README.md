# TWIT - Tensor Weighted Interpolative Transfer

Richard Keene September 2, 2019

# Concept of TWIT
The purpose of TWIT is to allow transfer of values from a source tensor to a destination tensor
with correct interpolation of values.  The source and destination do not have to have the same number of dimensions, 
e.g. len(src.shape) != len(dst.shape).
Also the range of indices do not have to match in count.  For example given two one dimensional tensors (vectors of values)
on could say copy from source range (2,7) to destination range (0,2) and use source to destination multipliers of (0.5, 0.9).  This will
copy source values at indicies 2,3,4,5,6,7 to destination indicies 0,1,2 (obviously 'scale' the data) and multiply 
source[2] by 0.5 to go to destination[0] and interpolate the multipler (weight) up to 0.9 for subsequent indicies.

### Energy Conservation
In the above example of scaling down the nieve approach would just sum in the source values multiplied by the interpolate
weights would result in a destination that is 6/3 or 2x 'brighter' than the source.
What TWIT does is maintain constant energy or brightness while scaling up or down.

Far more intuitive is the example of a 3D tensor which happens to be a color image, going to a destination tensor 
that is a 2D grey scale image.  We want to maintain brightness and the dimensions do not match.

This image scale operation might be a source image that is 300x400x3 to a destination that is 400x600.  Nothing matches.
Lets take the example of input to TWIT of 
*Note: One can use square brackets for clarity, twit does not care.*

([(0, 0, 1.0), (299, 399, 1.0)], [(0, 0, 1.0), (399, 599, 0.5)], [(0, 0, 1.0), (3, 0, 1.0)])

The parameters are pairs of tuples describing the start and end for each axis, outermost to innermost in order.
So (0, 0, 1.0), (299, 399, 1.0) indicates to use the sourece range 0 to 299 and destination range 0 to 399
and a weight range of 1.0 to 1.0 for the outermost axis, the height of the images.
Then (0, 0, 1.0), (399, 599, 0.5) says to use source 0 to 399, destination 0 to 599, and weights 1.0 down to 0.5
for the height to destination copy.  This will fade the image horizontally from 1.0 to medium dark 0.5.
The last pair (0, 0, 1.0), (3, 0, 1.0) says to use the RGB of the source to the single destination 'pixel'
and use the average (r + b + g)/3.0 as the value.

# Iteration
Once you have set up a twit, which is an iterator, it will return successive (sourceindex, destinationindex, weight) tuple sets.
In the image example above one iteration might return
((50, 67, 1.0), (100, 150, 0.7), (1, 0, 1.0)) I made thos numbers up, but they indicate:

destination[67, 150, 0] += source[50, 100, 1] * (1.0 * 0.7 * 1.0)

If you have loaded a color image into source, and destination starts as all zeros, then you get
a destination image stretched tall, and fading dimmer to the right.
Notice that the image does not have to be range 0 to 255, nor even ranges 0.0 to 1.0 pixels.  It can be any range
and the ranges only have meaning to your program.

One can also sepcify only a sub range in any dimension.  It does not have to be 0 to N. This will crop and 
scale to the destination cropped area.

Sidenote: The old fashioned way to the above is to use PIL and make an image from the source, scale it, do the fade,
then convert back to a tensor.  Ugly solution.

# Reuse of twit results
If you are going to use the twit result repeatedly you should make the iterator, generate all the tuple triplets
once and then reuse the cached array.

# Why TWIT?
The motivation for TWIT is to support the development of the SRS cognition algorithm and Cognate.  The system has
lots of concept maps that are N dimensional tensors represent some concept space.  They are neurons and they are
connections between the concept maps.  Every system tick we do a twit transfer (using cached twit iterator values)
to transform data.  The fact this library can als be used for image scaling in tensors is a side effect.

# Range strings
There is also a range definition string format used for Neural net concept map editing and viewing.
Input s is like "{5,17} <2,3> [-0.1, 0.9]" (see doc string for the module)
Format is ((srcstart, srcend), (dststart, dstend), (weightstart, weightend)).
src, dst, etc are to determine the ranges. The ..._shape_idx are which dimension of the array to use
for defaults.
Both indicies and weights can be revered to indicate a reversed interpolation along that axis.
Indexes are incluse "start value to end value", so {5, 9} means 5,6,7,8,9. 
This is in support of editors for people to easily enter ranges and weights for neural nets.
(See Cognate and NuTank)

# Helper Functions
TWIT includes some static functions to help do things.

apply_twit(twt, t1, t2) will iterate the twit twt and do the copy and multiplies from t1 to t2.  If t1 and t2 are not the
same numbe rof dimensions it will vreate the appropriate view and then do the work.  Tyhere is a clear destination flag to 
zero out the destination before iteration.  t2 MUST be a array style tensor since it gets written to in-place. You can pass
in an optional twit cache.

make_twit_cache(twt) Simply fills a list with the triples from the twit iterator.

A little helper function if you need it, match_tensor_shape_lengths, will make the views and return them given a t1 and t2.

# Tests
There is a twit_test.py file with all the unit tests.

# Examples
There is a image_scale.py test file for an example.  It is not very generic.
