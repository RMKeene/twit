"""
A Tensor Weighted Interpolated Transfer.
Algorithms by Richard Keene 7/2019. 
Copyright same as python: https://www.python.org/doc/copyright/ 2019

We have a source numpy.array, and a destination numpy.array.
They may or may not have the same number of dimensions and may or may not have the same 
size along any dimension.

We want to connect the source to the destination from some start and end index along each dimension of the source to
some start and end indicies on the destination.  The range of the source along an axis may or may not match the range along some axis of
the destination. 

We have a nomencalture where numbers in square brackets are the weights, curly brackets are the
source indicies and angle brackets are the destination indicies.

So "{5,17} <2,3> [-0.1, 0.9]" means source range index from 5 to 17 (inclusive), 2 to 3 in the destination, and a weight range from -0.1 to +0.9.
Weight interpolation is linear from start to end across the index ranges.
In the brackets can be two numbers and a comma, or just one number to indicate start and end, or no numbers.
If the index range is empty it indicates the entire range along the array axis.  An empty weight range is default 1.0.

Reversed number (first is higher than second) indicates a reverse range to interpolate.

Whichever index range is the largest determines how many steps along both indexes we generate.
The interpolater (iterator) generates three element tuples that are (srcidx, dstidx, weight) in a series. 

"""

import numpy as np
from collections import defaultdict

def tryParseInt(value, default_value=0):
    if value is None:
        return default_value, False
    try:
        return int(value), True
    except ValueError:
        return default_value, False


def tryParseFloat(value, default_value=0.0):
    if value is None:
        return default_value, False
    try:
        return float(value), True
    except ValueError:
        return default_value, False


def split_strip(s: str, sep=' '):
    """ 
    Split s into parts on delimiter, then strip the sub strings and remove any blanks.
    Never returns None.
    Returns an array of the sub strings.  The returned array my be empty.
    """
    if s is None:
        return []
    parts = s.split(sep=sep)
    ret = []
    if parts is not None:
        for ss in parts:
            sss = ss.strip()
            if len(sss) > 0:
                ret.append(sss)
    return ret


def twit_interp(range:tuple, v: tuple, idx: int):
    """ 
    Return the interpolatesd value from v[0] to v[1] along range at idx as a float.
    Range is a two entry tuple of float or integer.
    If range[0] equals range[1] then returns v1.  Both or either of range and v1 to v2 can be negative distances.
    Does not check that idx is in the range.  In which case it can return values outside the v[0] to v[1] range.
    Range and idx do not have to be integers.

    Implemented as:

    rspan = range[1] - range[0]
    if rspan == 0:
        return v[0]
    vspan = v[1] - v[0]
    return v[0] + (v[1] - v[0]) * (idx - range[0]) / rspan
    """
    rspan = range[1] - range[0]
    if rspan == 0:
        return v[0]
    vspan = v[1] - v[0]
    return v[0] + (v[1] - v[0]) * (idx - range[0]) / rspan



def outside_range(range: tuple, idx: int):
    if range[0] <= range[1]:
        return idx < range[0] or idx > range[1]
    return idx < range[1] or idx > range[0]


def twit_str_to_ranges(src, src_shape_idx, dst, dst_shape_idx, s: str) -> tuple:
    """ Converts a TWIT string for a single dimension to three pairs in a tuple. 
    Input s is like "{5,17} <2,3> [-0.1, 0.9]" (see doc string for the module)
    Format is ((srcstart, srcend), (dststart, dstend), (weightstart, weightend)).
    src, dst, etc are to determine the ranges. The ..._shape_idx are which dimension of the array to use
    for defaults.

    Both indicies and weights can be revered to indicate a reversed interpolation along that axis.

    Indexes are incluse "start value to end value", so {5, 9} means 5,6,7,8,9. 

    Returns tuple (None, reasonstring) if parsing fails.
    """
    assert(src is not None)
    assert(dst is not None)
    assert(src_shape_idx > 0 and src_shape_idx < len(src.shape))
    assert(dst_shape_idx > 0 and dst_shape_idx < len(dst.shape))
    
    # The defaults
    src_low_idx = 0
    src_hi_idx = src.shape[src_shape_idx] - 1
    dst_low_idx = 0
    dst_hi_idx = dst.shape[dst_shape_idx] - 1
    weight_low = 1.0
    weight_hi = 1.0

    return twit_string_to_ranges_internal(src_low_idx, src_hi_idx, dst_low_idx, dst_hi_idx, weight_low, weight_hi, s)


def parse_single_range_part(low, hi, s, leading, trailing, try_parse_func, enforce_low_hi=False):
    """
    In s expect some bracketed number pair. s is like "{5,17} <2,3> [-0.1, 0.9]"
    and we mayu want what is between { and }.
    low is the default low value.
    hi is the default hi value.
    leading and trailing are the brackets, like "{" and "}".
    enforce_low_hi fails if the result numers are outside the low or hi limits.

    returns a two number tuple on success, else (None, None) on error.  
    So 
        0, 20, "{", "}", tryParseInt, "{5,17} <2,3> [-0.1, 0.9]", True
    returns (5,17)
    See __main__ at the end of this file for examples.

    try_parse_func is the tryParseInt or tryParseFloat function. Must be like

    def tryParseInt(value, default_value = 0):
        if value is None:
            return default_value, False
        try:
            return int(value), True
        except ValueError:
            return default_value, False


    def tryParseFloat(value, default_value = 0.0):
        if value is None:
            return default_value, False
        try:
            return float(value), True
        except ValueError:
            return default_value, False
    """

    lowin = low
    hiin = hi
    iss = s.find(leading)
    ise = s.find(trailing)
    # Missmatched or out of order brackets?
    if (iss >= 0 and (ise == -1 or ise < iss)) or (iss == -1 and ise >= 0):
        return (None, None)
    if iss >= 0:
        parts = split_strip(s[iss + 1:ise], sep=',')
        if len(parts) > 0:
            low = try_parse_func(parts[0], low)
            if low[1] is False:
                return (None, None)
            low = low[0]
        if len(parts) > 1:
            hi = try_parse_func(parts[1], hi)
            if hi[1] is False:
                return (None, None)
            hi = hi[0]

    if enforce_low_hi:
        if low < lowin or hi < lowin or low > hiin or hi > hiin:
            return (None, None)
    return (low, hi)


def twit_string_to_ranges_internal(src_low_default, src_hi_default, dst_low_default, dst_hi_default, weight_low_default, weight_hi_default, s: str) -> tuple:
    """ 
    Converts a TWIT string for a single dimension to three pairs in a tuple. 
    Input s is like "{5,17} <2,3> [-0.1, 0.9]" (see doc string for the module)
    Format is ((srcstart, srcend), (dststart, dstend), (weightstart, weightend)).
    src, src_shape_idx etc. are to determine the defaults.
    The src and dst low and high are the absolute limits of the src and dest idx range.
    The low must be less than or equal to the hi.

    This is the version that does not need a numpy array pair where you specify the defaults directly.

    Returns tuple (None, reasonstring) if parsing fails.
    """

    assert(src_low_default >= 0 and src_low_default <= src_hi_default)
    assert(dst_low_default >= 0 and dst_low_default <= dst_hi_default)

    src_low_idx = src_low_default
    src_hi_idx = src_hi_default
    dst_low_idx = dst_low_default
    dst_hi_idx = dst_hi_default
    weight_low = weight_low_default
    weight_hi = weight_hi_default

    if s is None:
        s = ""

    src_low_idx, src_hi_idx = parse_single_range_part(src_low_idx, src_hi_idx, s, "{", "}", tryParseInt, True)
    if src_low_idx is None:
        return (None, 'Invalid source range')
    dst_low_idx, dst_hi_idx = parse_single_range_part(dst_low_idx, dst_hi_idx, s, "<", ">", tryParseInt, True)
    if dst_low_idx is None:
        return (None, 'Invalid destination range')
    weight_low, weight_hi = parse_single_range_part(weight_low, weight_hi, s, "[", "]", tryParseFloat, False)
    if weight_low is None:
        return (None, 'Invalid destination range')
       
    return ((src_low_idx, src_hi_idx), (dst_low_idx, dst_hi_idx), (weight_low, weight_hi))


class twit_fan_enum:
    # Source and destination span are the same.
    FAN_SAME = 0
    # Source index span is greater than destination span.
    FAN_IN = 1
    # Destination span is greater.
    FAN_OUT = 2


def find_range_series_multipliers(narrow_range, wide_range, narrow_idx):
    """
    Give a narrow_range, like (0,4) being the range indixes 0,1,2,3,4
    and a wider_range like (2,9) beign 2,3,4,5,6,7,8,9
    and an index in the narrow range, like 1:
    Make a tuple of pairs of (idx, weight) that is the connections from the narrow
    to the wide for narrow idx 1 in an interplative fashion across integer indicies.
    Return enties are not guaranteed to be in any particular order.
    Return weights
    """
    if narrow_range is None or wide_range is None:
        raise Exception("find_range_series_multipliers: wide and/or narrow range is None.")
    if len(narrow_range) != 2:
        raise Exception("find_range_series_multipliers: narrow_range must have exactly two elements.")
    if len(wide_range) != 2:
        raise Exception("find_range_series_multipliers: wide_range must have exactly two elements.")
    if narrow_idx < min(narrow_range[0], narrow_range[1]) or narrow_idx > max(narrow_range[0], narrow_range[1]):
        raise Exception("find_range_series_multipliers: narrow_idx is out of range.  Must be in the narrow_range (inclusive).")

    # Force narrow and wide ranges to be in order.  At this low level it does
    # not matter which order we sequence the return values.
    if narrow_range[0] > narrow_range[1]:
        narrow_range = (narrow_range[1], narrow_range[0])
    if wide_range[0] > wide_range[1]:
        wide_range = (wide_range[1], wide_range[0])

    if narrow_range[0] < 0 or wide_range[0] < 0:
        raise Exception("find_range_series_multipliers: Negative range indicies.")

    narrow_span = narrow_range[-1] - narrow_range[0] + 1
    wide_span = wide_range[-1] - wide_range[0] + 1
    if narrow_span >= wide_span:
        raise Exception("find_range_series_multipliers: Wide range must be wider than narrow_range.")
    wspan = wide_span - 1

    # Generate the fractional values.
    ret = []
    sum = 0.0
    narrow_relidx = narrow_idx - narrow_range[0]
    narrow_div = narrow_span - 1
    if narrow_div > 0:
        narrow_idx_frac_low = max(0.0, (narrow_relidx - 1) / narrow_div)
        narrow_idx_frac = narrow_relidx / narrow_div
        narrow_idx_frac_hi = min(1.0, (narrow_relidx + 1) / narrow_div)

        wide_idx_low = max(0, int(narrow_idx_frac_low * wspan))
        wide_idx_mid = narrow_idx_frac * wspan
        wide_idx_hi = min(wspan, int(narrow_idx_frac_hi * wspan))
        wide_half_span = wspan / narrow_div
        for i in range(int(wide_idx_low), int(wide_idx_hi) + 1):
            frac = 1.0 - abs((i - wide_idx_mid) / wide_half_span)
            if frac > 0.0:
                ret.append((i + wide_range[0], frac))
            sum += frac
    else: # One to Many
        frac = 1.0 / wide_span
        for i in range(int(wide_range[0]), int(wide_range[1]) + 1):
            ret.append((i, frac))
            sum += frac

    # Normalize so sum is 1.0
    for i in range(len(ret)):
        ret[i] = (int(ret[i][0]), ret[i][1] / sum)

    # Weights of ret will always sum to 1.0 (They are normalized)
    return tuple(ret)


class twit_single_axis_discrete:
    """
    Encapsulates a twit_single_axis and iterates a series of integet indices and float weights.
    Currently only works with axis_range like ((start_source_idx, start_dest_idx, start_weight), (end_source_idx, end_dest_idx, end_weight)).
    This provides the fan in or fan out to discrete tensors.
    For example a single iteration of a twit_single_axis  that is 10 source to 4 destination,
    it is fan-in source to destination so a single iterationm from the twit_single_axis
    might be (1.0, 1.7, 0.3) being (source idx, dest idx, weight).  
    """

    def __init__(self, axis_range: tuple):
        # TODO - Make this more general.  Maybe handle >= 3 elements so
        # multiple dependent values
        # can be interpolated.
        if len(axis_range) != 2 or len(axis_range[0]) != 3 or len(axis_range[1]) != 3:
           raise Exception("twit_single_axis_discrete only applies to a source and destination ranges and a weight.")
        self.next_returns = []
        self._ended_ = False
        # Our spans are inclusive so the + 1 on the end here.
        self.src_range = (axis_range[0][0], axis_range[1][0])
        self.dst_range = (axis_range[0][1], axis_range[1][1])
        self.weight_range = (axis_range[0][2], axis_range[1][2])
        self.input_span = abs(self.src_range[0] - self.src_range[1]) + 1
        self.output_span = abs(self.dst_range[0] - self.dst_range[1]) + 1
        # Weight span is signed and is the simple difference.
        self.weight_span = self.weight_range[1] - self.weight_range[0]
        if self.input_span == self.output_span:
            self.fan = twit_fan_enum.FAN_SAME
        elif self.input_span > self.output_span:
            self.fan = twit_fan_enum.FAN_IN
        else:
            self.fan = twit_fan_enum.FAN_OUT
        self.src_idx = self.src_range[0]
        self.dst_idx = self.dst_range[0]
        self.src_inc = 1
        if self.src_range[1] < self.src_range[0]:
            self.src_inc = -1
        self.dst_inc = 1
        if self.dst_range[1] < self.dst_range[0]:
            self.dst_inc = -1
        self.value_cache = []


    def generate_value_cache(self):
        self.value_cache.clear()
        if self.fan == twit_fan_enum.FAN_SAME:
            dsti = self.dst_range[0]
            for srci in range(int(self.src_range[0]), int(self.src_range[1]) + self.src_inc, self.src_inc):
                self.value_cache.append((srci, dsti, twit_interp(self.src_range, self.weight_range, srci)))
                dsti += self.dst_inc
            # Normalization not needed for one to one.
        elif self.fan == twit_fan_enum.FAN_IN:
            for dsti in range(int(self.dst_range[0]), int(self.dst_range[1]) + self.dst_inc, self.dst_inc):
                splits = find_range_series_multipliers(self.dst_range, self.src_range, dsti)
                for sp in splits:
                    # We use an array (rather than a tuple) so it is in-place modifiable.
                    self.value_cache.append([sp[0], dsti, sp[1]])
            # Normalize
            dstsums = defaultdict(float)
            for v in self.value_cache:
                di = v[1]
                dstsums[di] = dstsums[di] + v[2]
            for x in self.value_cache:
                x[2] = x[2] * twit_interp(self.src_range, self.weight_range, x[0]) / dstsums[x[1]]
            pass
        else: # Fan out
            for srci in range(int(self.src_range[0]), int(self.src_range[1]) + self.src_inc, self.src_inc):
                splits = find_range_series_multipliers(self.src_range, self.dst_range, srci)
                for sp in splits:
                    self.value_cache.append([srci, sp[0], sp[1]])
            # Normalize
            dstsums = defaultdict(float)
            srcsums = defaultdict(float)
            for v in self.value_cache:
                si = v[0]
                srcsums[si] = srcsums[si] + v[2]
                di = v[1]
                dstsums[di] = dstsums[di] + v[2]
            for x in self.value_cache:
                src_to_dst_ratio = srcsums[x[0]] / dstsums[x[1]]
                x[2] = x[2] * twit_interp(self.dst_range, self.weight_range, x[1]) / dstsums[x[1]]
            pass
        pass

    
    def __iter__(self):
        self.reset()
        return self


    def __next__(self):
        if len(self.value_cache) == 0:    
            raise StopIteration
        return tuple(self.value_cache.pop(0))


    def reset(self):
        self.generate_value_cache()
        self.next_returns = []
        self._ended_ = False


    def is_at_end():
        return self._ended_


class twit:
    """
    A Tensor Weighted Interpolative Transfer iterator multi dimensional. See top of this twit.py file.
    __next__() returns a tuple of tuples one for each axis.
    being
    ((SrcNidx, DstNidx, WeightNidx) for each axis down to axis 0) assuming we are 
    doing neural net weighted connections between concept maps.
    Works for 1 to N dimensions.

    If you are generating the connections between two tensors of diffent count of axis, 
    e.g. len(t1.shape) != len(t2.shape) then have the index range and weight for the shorter number of exes be (0, 0, 1.0)

    """

    def __init__(self, ranges: tuple):
        """
        ranges is a tuple of three tuples, like 
        (((srcstart, srcend), (dststart, dstend), (weightstart, weightend)), ... each axis down to axis 0 ).
        The src and dst are int, weights are float.
        """
        self.ranges = ranges
        # The iterators and axes are in the same order as python axis
        # so innermost (x) most rapidly changing last.
        self.iterators = []
        self.lastNextValues = []
        self.axes = []
        for i in range(len(ranges)):
            t = ranges[i]
            # zip here is doing a transpose of the tuple.
            # https://gist.github.com/CMCDragonkai/0cd2cc8c0aa7fd5eeec3955052dfd344
            tt = tuple(zip(*t))
            a = twit_single_axis_discrete(tt)
            self.axes.append(a)
            i = iter(a)
            self.iterators.append(i)
            self.lastNextValues.append(None)
        self.reset()
        pass
    
    
    def __iter__(self):
        self.reset()
        return self


    def make_current_value_tuples(self):
        """
        Concatenate all the currentValues of the list of iterators.
        This is the currentValue of this overall N-Dimensional iterator.
        Returns nothing.  self.currentValue is updated.
        If any iterators are at None values then next() gets called for that iterator.
        So... this method can throw STOP_ITERRATION.
        """
        currentValue = ()
        for i in range(len(self.iterators)):
            if self.lastNextValues[i] is None:
                self.lastNextValues[i] = next(self.iterators[i])
            currentValue += (self.lastNextValues[i],)
        return currentValue


    def __next__(self):
        for i in range(len(self.iterators) - 1, -1, -1):
            threw = False
            hit = False
            try:
                self.lastNextValues[i] = next(self.iterators[i])
            except StopIteration:
                threw = True
            if threw is False:
                hit = True
                break
            self.iterators[i].reset()
            self.lastNextValues[i] = None
        if hit == False:
            self.AtEnd = True
            raise StopIteration()

        cv = self.make_current_value_tuples()
        return cv


    def reset(self):
        for i in range(len(self.iterators)):
            self.iterators[i].reset()
            self.lastNextValues[i] = None
    pass


def match_tensor_shape_lengths(t1, t2):
    """
    Ensure t1 and t2 have the same count of axes with the largest count of axes 
    as the desired size.  If t1 or t2 is shorter in axis count, add new axes of dimension 1.
    Returns the correctly sized two tensors.  Will make a view if not correct. Else uses the passed in 
    t1 and t2 ans the return vlaues (t1, t2).

    This is used to get two tensor shapes able to be used by twit with the same number of axes each.
    """
    t1L = len(t1.shape)
    t2L = len(t2.shape)
    if t1L < t2L:
        t1s = t1.shape
        while len(t1s) < t2L:
            t1s = (1,) + t1s
        t1 = np.reshape(t1, t1s)
    elif t2L < t1L:
        t2s = t2.shape
        while len(t2s) < t1L:
            t2s = (1,) + t2s
        t2 = np.reshape(t2, t2s)
    return (t1, t2)


def make_twit_cache(twt: twit):
    assert(isinstance(twt, twit))
    cache = []
    for t in twt:
        cache.append(t)
    return cache


def apply_twit(t1, t2, twt: twit = None, cache: list = None, preclear: bool = True):
    """
    Apply the twit transfer from t1 to t2.  Makes view if needed to get shapes compatible.
    preclear True will zero out t2 in the possibly sub region of the twit destination. 
    One of twt or cache must be valid but not both.
    """
    if twt is not None:
        cache = make_twit_cache(twt)
    elif cache is None:
        raise AttributeError("one of twt or cache MUST be valid.")

    t1, t2 = match_tensor_shape_lengths(t1, t2)
    if preclear:
        for t in cache:
            z = list(zip(*t))
            t2[z[1]] = 0.0
    for t in cache:
        z = list(zip(*t))
        t2[z[1]] += t1[z[0]] * np.prod(z[2])
    pass


if __name__ == '__main__':
    # See twit_test.py for unit tests.
    pass


