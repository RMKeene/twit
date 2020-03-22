"""
A Tensor Weighted Interpolated Transfer.
Algorithms by Richard Keene 7/2019. 
Copyright same as python: https:#www.python.org/doc/copyright/ 2019

We have a source numpy.array, and a destination numpy.array.
They may or may not have the same number of dimensions and may or may not have the same 
size along any dimension.

We want to connect the source to the destination from some start and end index along each dimension of the source to
some start and end indicies on the destination.  The range of the source along an axis may or may not match the range
along some axis of the destination.

We have a nomenclature where numbers in square brackets are the weights, curly brackets are the
source indices and angle brackets are the destination indices.

So "{5,17} <2,3> [-0.1, 0.9]" means source range index from 5 to 17 (inclusive), 2 to 3 in the destination,
and a weight range from -0.1 to +0.9. Weight interpolation is linear from start to end across the index ranges. In
the brackets can be two numbers and a comma, or just one number to indicate start and end, or no numbers. If the
index range is empty it indicates the entire range along the array axis.  An empty weight range is default 1.0.

Reversed number (first is higher than second) indicates a reverse range to interpolate.

Whichever index range is the largest determines how many steps along both indexes we generate.
The interpolater (iterator) generates three element tuples that are (srcidx, dstidx, weight) in a series.

Why? Because SRS, Cognate and NuTank are doing neural net as concept maps and we transfer or 'wire'
connections between concept maps. Also lets one do image scale and crop on tensors directly
without having to resort to PIL scale etc. and conversion back and forth.
"""

from typing import Tuple, Optional, AnyStr, List, Sequence, Union
import numpy as np

# Source and destination span are the same.
TWIT_FAN_SAME = 0
# Source index span is greater than destination span.
TWIT_FAN_IN = 1
# Destination span is greater.
TWIT_FAN_OUT = 2


class RangeSeries:
    def __init__(self, length: int):
        self.length = length
        self.idxs = np.zeros(length)
        self.values = np.zeros(length)


class TwitSingleAxis:
    def __init__(self, length: int):
        self.length = length
        self.srcidxs = np.zeros(length, dtype=np.int64)
        self.dstidxs = np.zeros(length, dtype=np.int64)
        self.weights = np.zeros(length, dtype=np.float)


class TwitMultiAxis:
    def __init__(self, length: int):
        self.length = length
        self.axs = []


class TwitException(Exception):
    """Base class for exceptions in this module."""
    pass


class TwitError(Exception):
    """Base class for exceptions in this module."""
    pass


class ParameterError(TwitError):
    """Raised when an method is given a parameter that is invalid.

    Attributes:
        message -- explanation of why the specific transition is not allowed
    """

    def __init__(self, message):
        self.message = message


def try_parse_int(value: Optional[AnyStr] = None, default_value: float = 0) -> Tuple[float, bool]:
    """
    Try to parse a string value to an int.  Returns the value and True
    e.g.  tryParseInt("42", 7) returns (42, True)
    tryParseInt("abcdef", 7) returns (7, False)
    See twit_test.py
    """
    try:
        return int(value), True
    except (ValueError, TypeError):
        return default_value, False


def try_parse_float(value: Optional[AnyStr] = None, default_value: float = 0) -> Tuple[float, bool]:
    """
    Try to parse a string value to an float.  Returns the value and True
    e.g.  tryParseInt("42.42", 7.3) returns (42.42, True)
    tryParseInt("abcdef", 7.3) returns (7.3, False)
    See twit_test.py
    """
    try:
        return float(value), True
    except (ValueError, TypeError):
        return default_value, False


def split_strip(s: Optional[AnyStr], sep=' ') -> List:
    """
    Split s into parts on delimiter, then strip the sub strings and remove any blanks.
    Never returns None.
    Returns an array of the sub strings.  The returned array my be empty.
    See twit_test.py for examples.
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


# Returns either +1 or -1, if x is 0 then returns +1
def twit_sign(x: float) -> int:
    if x < 0:
        return -1
    else:
        return 1


def match_tensor_shape_lengths(t1: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure t1 and t2 have the same count of axes with the largest count of axes
    as the desired size.  If t1 or t2 is shorter in axis count, add new axes of dimension 1 to the
    left of the shape.
    Returns the correctly sized two tensors.  Will make a view if not correct. Else uses the passed in
    t1 and t2 as the return vlaues (t1, t2). See twit_test.py

    This is used to get two tensor shapes able to be used by twit with the same number of axes each.
    """
    t1len = len(t1.shape)
    t2len = len(t2.shape)
    if t1len < t2len:
        t1s = t1.shape
        while len(t1s) < t2len:
            t1s = (1,) + t1s
        t1 = np.reshape(t1, t1s)
    elif t2len < t1len:
        t2s = t2.shape
        while len(t2s) < t1len:
            t2s = (1,) + t2s
        t2 = np.reshape(t2, t2s)
    return t1, t2


def match_shape_lengths(s1: Tuple, s2: Tuple) -> Tuple:
    """
    Ensure s1 and s2 have the same count of axes with the largest count of axes
    as the desired size.  If s1 or s2 is shorter in axis count, add new axes of dimension 1 to the
    left of the shape.
    Returns the correctly sized two shapes. See twit_test.py

    This is used to get two tensor shapes able to be used by twit with the same number of axes each.
    """
    if len(s2) == 0:
        raise AttributeError("Tensor destination shape can not be length 0, nowhere to put the results!")
    while len(s1) < len(s2):
        s1 = (1,) + s1
    while len(s2) < len(s1):
        s2 = (1,) + s2
    return s1, s2


def gen_twit_param_from_shapes(sh1: Tuple, sh2: Tuple,
                               weight_axis: int = -1,
                               weight_range: tuple = (1.0, 1.0)) -> Tuple[int, Sequence[int], Sequence[float]]:
    """
    Given two tensor shapes, generate the twit specification tuple to transfer all of t1 to t2.
    Assumes all weights are 1.0.  Can override one axis of weights.

    Return is a tuple like (2, [0, 3, 0, 4, 0, 2, 0, 3], [0.0, 1.0, 1.0, 1.0]) given that
    t1 shape is 3,2 and t2 shape is 4,3 and weight_range is 0.0, 1.0 and weight_axis is 0.
    """
    sh1, sh2 = match_shape_lengths(sh1, sh2)
    idx_series = np.zeros(len(sh1) * 4, dtype=np.int64)
    weight_series = np.zeros(len(sh1) * 2, dtype=np.float)
    ret = (len(sh1), idx_series, weight_series)
    for i in range(len(sh1)):
        i4 = i * 4
        i2 = i * 2
        idx_series[i4] = 0
        idx_series[i4 + 1] = sh1[i] - 1
        idx_series[i4 + 2] = 0
        idx_series[i4 + 3] = sh2[i] - 1
        if i == weight_axis:
            weight_series[i2] = weight_range[0]
            weight_series[i2 + 1] = weight_range[1]
        else:
            weight_series[i2] = 1.0
            weight_series[i2 + 1] = 1.0
    return ret


def gen_twit_param(t1: np.ndarray, t2: np.ndarray, weight_axis: int = -1, weight_range: tuple = (1.0, 1.0)) -> \
        Tuple[int, Sequence[int], Sequence[float]]:
    """
    Generate and return the twit specification tuples that will
    transfer all of t1 to t2 along all axes with weight 1.0.
    See twit_test.py
    Note that shape lengths are left inclusive, so shape 3 is indices 0,1,2 but
    twit convention is inclusive so the range (0,3) is 0,1,2,3.  Don't get confused.
    Save lots of mistakes by using twit.py as as an index generatore like

    python twit.py (4,6,3) (7,8)  and it will generate the params.

    # Scale image to match.
    t2: np.ndarray = np.zeros(self.client_concept_map.get_shape())
    twit_params: Tuple[int, Sequence[int], Sequence[float]] = twit.gen_twit_param(self.np_im, t2)
    twt = twit.compute_twit_multi_dimension(*twit_params)
    twitc.apply_twit(twt, self.np_im, t2, 0)

    """
    return gen_twit_param_from_shapes(t1.shape, t2.shape, weight_axis=weight_axis, weight_range=weight_range)


def twit_interp(range_start: int, range_end: int, value_start: float, value_end: float, idx: int) -> float:
    """
    Return the interpolated value from value_start to value_end along range at idx as a float. Range is a two entry
    tuple of float or integer. If range_start equals range_end then returns value_start.  Both or either of range and
    v1 to v2 can be negative distances. Does not check that idx is in the range.  In which case it can return values
    outside the v[0] to v[1] range. Range and idx do not have to be integers.
    """
    rspan = range_end - range_start
    if rspan == 0:
        return value_start
    return value_start + (value_end - value_start) * (idx - range_start) / rspan


def outside_range(start: int, end: int, idx: int) -> bool:
    """double
    True if idx is not between start and end inclusive.
    """
    if start <= end:
        return idx < start or idx > end
    return idx < end or idx > start


def twit_str_to_multi_params(
        src_shape: Union[Tuple, np.ndarray], dst_shape: Union[Tuple, np.ndarray],
        s: AnyStr) -> Tuple:
    """ Takes like  src shape (5, 4) and dst shape (8, 3, 2) and
    s of '{0_4, 1_3} <0_8, 0_3, 2_3> [1_1, 0_1, 0.5_0.6]'  and creates the
    three tuples needed by compute_twit_multi_dimension. Does bounds checking and defaults values.
    You can then pass the tuples in as the params with *t
    The above example will return (3, [0, 0, 0, 4, 1, 3, 0, 8, 0, 3, 2, 3], [1.0, 1.0, 0.0, 1.0, 0.5, 0.6])
    Being (axis_count,
        [srclow[0], srcHi[0], srclow[1], srcHi[1], srclow[2], srcHi[2],
         dstlow[0], dstHi[0], dstlow[1], dstHi[1], dstlow[1], dstHi[1]],
        [wLow[0], wHi[0], wLow[1], wHi[1], wLow[2], wHi[2]])
    Note that ranges like 0_4 mean 0,1,2,3,4 so are inclusive.
    Errors cause a return of (False, "error reason")
    Note src_shape and dst_shape can be ndarray, Tuple (being the concept map shape).
    """

    if isinstance(src_shape, np.ndarray):
        src_shape = src_shape.shape
    elif isinstance(src_shape, Tuple):
        pass
    else:
        assert False, "twit_str_to_multi_params: Expects src of Tuple or np.ndarray"

    if isinstance(dst_shape, np.ndarray):
        dst_shape = dst_shape.shape
    elif isinstance(dst_shape, Tuple):
        pass
    else:
        assert False, "twit_str_to_multi_params: Expects dst of Tuple or np.ndarray"
    # get the three parts if present
    src_clause = ""
    dst_clause = ""
    w_clause = ""

    src_shape, dst_shape = match_shape_lengths(src_shape, dst_shape)

    index_params = [0] * (len(src_shape) * 4)
    weight_params = [1.0] * (len(src_shape) * 2)

    # SRC
    src_clause_start_idx = s.find("{")
    if src_clause_start_idx >= 0:
        src_clause_end_idx = s.find("}")
        if src_clause_end_idx > src_clause_start_idx:
            src_clause = s[src_clause_start_idx + 1: src_clause_end_idx]
        else:
            return False, "Invalid source, no closing }"

    src_clauses = split_strip(src_clause, sep=",")
    while len(src_clauses) < len(src_shape):
        src_clauses.insert(0, "")
    for i in range(len(src_shape)):
        inner_parts = split_strip(src_clauses[i], sep='_')
        if len(inner_parts) == 0:
            index_params[i * 2] = 0
            index_params[i * 2 + 1] = src_shape[i] - 1
        elif len(inner_parts) == 1:
            v, b = try_parse_int(inner_parts[0])
            if b and 0 <= v < src_shape[i]:
                index_params[i * 2] = int(v)
                index_params[i * 2 + 1] = int(v)
            else:
                return False, "Invalid source index for dimension %d" % (i,)
        else:
            v, b = try_parse_int(inner_parts[0])
            if b and 0 <= v < src_shape[i]:
                index_params[i * 2] = int(v)
            else:
                return False, "Invalid source index for dimension %d" % (i,)
            v, b = try_parse_int(inner_parts[1])
            if b and 0 <= v < src_shape[i]:
                index_params[i * 2 + 1] = int(v)
            else:
                return False, "Invalid source index for dimension %d" % (i,)

    dst_clause_start_idx = s.find("<")
    if dst_clause_start_idx >= 0:
        dst_clause_end_idx = s.find(">")
        if dst_clause_end_idx > dst_clause_start_idx:
            dst_clause = s[dst_clause_start_idx + 1: dst_clause_end_idx]
        else:
            return False, "Invalid destination, no closing >"

    dst_clauses = split_strip(dst_clause, sep=",")
    while len(dst_clauses) < len(dst_shape):
        dst_clauses.insert(0, "")
    index_params_offset = len(dst_shape) * 2
    for i in range(len(dst_shape)):
        inner_parts = split_strip(dst_clauses[i], sep='_')
        if len(inner_parts) == 0:
            index_params[index_params_offset + i * 2] = 0
            index_params[index_params_offset + i * 2 + 1] = dst_shape[i] - 1
        elif len(inner_parts) == 1:
            v, b = try_parse_int(inner_parts[0])
            if b and 0 <= v < dst_shape[i]:
                index_params[index_params_offset + i * 2] = int(v)
                index_params[index_params_offset + i * 2 + 1] = int(v)
            else:
                return False, "Invalid destination index for dimension %d" % (i,)
        else:
            v, b = try_parse_int(inner_parts[0])
            if b and 0 <= v < dst_shape[i]:
                index_params[index_params_offset + i * 2] = int(v)
            else:
                return False, "Invalid destination index for dimension %d" % (i,)
            v, b = try_parse_int(inner_parts[1])
            if b and 0 <= v < dst_shape[i]:
                index_params[index_params_offset + i * 2 + 1] = int(v)
            else:
                return False, "Invalid destination index for dimension %d" % (i,)

    w_clause_start_idx = s.find("[")
    if w_clause_start_idx >= 0:
        w_clause_end_idx = s.find("]")
        if w_clause_end_idx > w_clause_start_idx:
            w_clause = s[w_clause_start_idx + 1: w_clause_end_idx]
        else:
            return False, "Invalid weights, no closing ]"

    w_clauses = split_strip(w_clause, sep=",")
    while len(w_clauses) < len(src_shape):
        w_clauses.insert(0, "")
    for i in range(len(src_shape)):
        inner_parts = split_strip(w_clauses[i], sep='_')
        if len(inner_parts) == 0:
            weight_params[i * 2] = 1.0
            weight_params[i * 2 + 1] = 1.0
        elif len(inner_parts) == 1:
            v, b = try_parse_float(inner_parts[0])
            if b:
                weight_params[i * 2] = v
                weight_params[i * 2 + 1] = v
            else:
                return False, "Invalid weight index for dimension %d" % (i,)
        else:
            v, b = try_parse_float(inner_parts[0])
            if b:
                weight_params[i * 2] = v
            else:
                return False, "Invalid weight index for dimension %d" % (i,)
            v, b = try_parse_float(inner_parts[1])
            if b:
                weight_params[i * 2 + 1] = v
            else:
                return False, "Invalid weight index for dimension %d" % (i,)

    return len(src_shape), np.array(index_params, dtype=np.int64), np.array(weight_params, dtype=np.float64)


def format_float(f):
    if f - float(int(f)) == 0.0:
        return str(int(f))
    s = ("%20.15f" % (f,)).lstrip(" ").rstrip("0")
    return s


def twit_multi_params_to_str(n_dims: int, twit_i: Sequence[int], twit_w: Sequence[float]) -> str:
    src = "{"
    dst = "<"
    w = "["
    nd2 = n_dims * 2
    for i in range(n_dims):
        if i > 0:
            src += ", "
            dst += ", "
            w += ", "
        src += "%d_%d" % (twit_i[i * 2], twit_i[i * 2 + 1])
        dst += "%d_%d" % (twit_i[nd2 + i * 2], twit_i[nd2 + i * 2 + 1])
        w += "%s_%s" % (format_float(twit_w[i * 2]), format_float(twit_w[i * 2 + 1]))
    src += "} "
    dst += "> "
    w += "]"
    return src + dst + w


def twit_str_to_ranges(src: np.ndarray, src_shape_idx: int,
                       dst: np.ndarray, dst_shape_idx: int,
                       s: Optional[AnyStr]) -> tuple:
    """ Converts a TWIT string for a single dimension to three pairs in a tuple.
    Input s is like "{5,17} <2,3> [-0.1, 0.9]" (see doc string for the module)
    Return is ((srcstart, srcend), (dststart, dstend), (weightstart, weightend)).
    src, dst, etc are to determine the ranges. The ..._shape_idx are which dimension of the array to use
    for defaults.

    Both indices and weights can be revered to indicate a reversed interpolation along that axis.

    Indexes are inclusive "start value to end value", so {5, 9} means 5,6,7,8,9.

    Returns tuple (None, reason_string) if parsing fails.
    """
    assert (src is not None)
    assert (dst is not None)
    assert (0 < src_shape_idx < len(src.shape))
    assert (0 < dst_shape_idx < len(dst.shape))

    # The defaults
    src_low_idx = 0
    src_hi_idx = src.shape[src_shape_idx] - 1
    dst_low_idx = 0
    dst_hi_idx = dst.shape[dst_shape_idx] - 1
    weight_low = 1.0
    weight_hi = 1.0

    return twit_string_to_ranges_internal(src_low_idx, src_hi_idx, dst_low_idx, dst_hi_idx, weight_low, weight_hi, s)


def parse_single_range_part(low: float, hi: float, s: AnyStr,
                            leading='{', trailing='}', try_parse_func=try_parse_int,
                            enforce_low_hi=False) -> Tuple[Optional[float], Optional[float]]:
    """
    In s expect some bracketed number pairs. s is like "{5,17} <2,3> [-0.1, 0.9]"
    and we may want what is between { and }.
    low is the default low value.
    hi is the default hi value.
    leading and trailing are the brackets, like "{" and "}".
    enforce_low_hi fails if the result numbers are outside the low or hi limits.

    returns a two number tuple on success, else (None, None) on error.
    So
        0, 20, "{", "}", tryParseInt, "{5,17} <2,3> [-0.1, 0.9]", True
    returns (5,17)
    See twit_string_to_ranges_internal for examples.
    """
    low_in = low
    hi_in = hi
    iss = s.find(leading)
    ise = s.find(trailing)
    # Mismatched or out of order brackets?
    if (iss >= 0 and (ise == -1 or ise < iss)) or (iss == -1 and ise >= 0):
        return None, None
    if iss >= 0:
        parts = split_strip(s[iss + 1:ise], sep=',')
        if len(parts) > 0:
            low = try_parse_func(parts[0], low)
            if low[1] is False:
                return None, None
            low = low[0]
        if len(parts) > 1:
            hi = try_parse_func(parts[1], hi)
            if hi[1] is False:
                return None, None
            hi = hi[0]

    if enforce_low_hi:
        if low < low_in or hi < low_in or low > hi_in or hi > hi_in:
            return None, None
    return low, hi


def twit_string_to_ranges_internal(src_low_default: int, src_hi_default: int,
                                   dst_low_default: int, dst_hi_default: int,
                                   weight_low_default: float, weight_hi_default: float, s: Optional[AnyStr]) -> Tuple:
    """
    Converts a TWIT string for a single dimension to three pairs in a tuple.
    Input s is like "{5,17} <2,3> [-0.1, 0.9]" (see doc string for the module)
    Format is ((srcstart, srcend), (dststart, dstend), (weightstart, weightend)).
    src, src_shape_idx etc. are to determine the defaults.
    The src and dst low and high are the absolute limits of the src and dest idx range.
    The low must be less than or equal to the hi.

    This is the version that does not need a numpy array pair where you specify the defaults directly.

    Returns tuple (None, reason_string) if parsing fails.
    """

    assert (0 <= src_low_default <= src_hi_default)
    assert (0 <= dst_low_default <= dst_hi_default)

    src_low_idx = src_low_default
    src_hi_idx = src_hi_default
    dst_low_idx = dst_low_default
    dst_hi_idx = dst_hi_default
    weight_low = weight_low_default
    weight_hi = weight_hi_default

    if s is None:
        s = ""

    src_low_idx, src_hi_idx = parse_single_range_part(src_low_idx, src_hi_idx, s, "{", "}", try_parse_int, True)
    if src_low_idx is None:
        return None, 'Invalid source range'
    dst_low_idx, dst_hi_idx = parse_single_range_part(dst_low_idx, dst_hi_idx, s, "<", ">", try_parse_int, True)
    if dst_low_idx is None:
        return None, 'Invalid destination range'
    weight_low, weight_hi = parse_single_range_part(weight_low, weight_hi, s, "[", "]", try_parse_float, False)
    if weight_low is None:
        return None, 'Invalid weight range'
    return (src_low_idx, src_hi_idx), (dst_low_idx, dst_hi_idx), (weight_low, weight_hi)


def find_range_series_multipliers(
        narrow_range_start: int, narrow_range_end: int,
        wide_range_start: int, wide_range_end: int, narrow_idx: int) -> RangeSeries:
    """
    Give a narrow_range, like (0,4) being the range indixes 0,1,2,3,4
    and a wider_range like (2,9) beign 2,3,4,5,6,7,8,9
    and an index in the narrow range, like 1:
    Make a tuple of pairs of (idx, weight) that is the connections from the narrow
    to the wide for narrow idx 1 in an interplative fashion across integer indicies.
    Return enties are not guaranteed to be in any particular order.
    Return weights are float and sum to 1.0 and are fractional parts of the link from source to destination.
    Higher levels of code apply your specified weights passed to twit(...)

    This method is where all the hard algorithm work gets done. Here there be dragons.
    """
    if narrow_idx < min(narrow_range_start, narrow_range_end) or narrow_idx > max(narrow_range_start, narrow_range_end):
        raise TwitException(
            "find_range_series_multipliers: narrow_idx is out of range.  Must be in the narrow_range (inclusive).")

    # Force narrow and wide ranges to be in order.  At this low level it does
    # not matter which order we sequence the return values.
    if narrow_range_start > narrow_range_end:
        narrow_range_start, narrow_range_end = narrow_range_end, narrow_range_start
    if wide_range_start > wide_range_end:
        wide_range_start, wide_range_end = wide_range_end, wide_range_start

    if narrow_range_start < 0 or wide_range_start < 0:
        raise TwitException("find_range_series_multipliers: Negative range indicies.")

    narrow_span = narrow_range_end - narrow_range_start + 1
    wide_span = wide_range_end - wide_range_start + 1
    if narrow_span >= wide_span:
        raise TwitException("find_range_series_multipliers: Wide range must be wider than narrow_range.")
    wspan = wide_span - 1

    # Generate the fractional values.
    frac_sum = 0.0
    narrow_relidx = narrow_idx - narrow_range_start
    narrow_div = narrow_span - 1
    element_count = 0
    if narrow_div > 0:
        narrow_idx_frac_low = max(0.0, (narrow_relidx - 1) / narrow_div)
        narrow_idx_frac = narrow_relidx / narrow_div
        narrow_idx_frac_hi = min(1.0, (narrow_relidx + 1) / narrow_div)

        wide_idx_low = max(0, int(narrow_idx_frac_low * wspan))
        wide_idx_low_int = int(wide_idx_low)
        wide_idx_mid = narrow_idx_frac * wspan
        wide_idx_hi = min(wspan, int(narrow_idx_frac_hi * wspan))
        wide_half_span = wspan / narrow_div
        fracs = np.zeros(wide_idx_hi - wide_idx_low + 1)
        for i in range(wide_idx_low_int, int(wide_idx_hi) + 1):
            fracs[i - wide_idx_low_int] = 1.0 - abs((i - wide_idx_mid) / wide_half_span)
            if fracs[i - wide_idx_low_int] > 0.0:
                element_count += 1
        ret = RangeSeries(element_count)
        idx = 0
        for i in range(wide_idx_low_int, int(wide_idx_hi) + 1):
            frac = fracs[i - wide_idx_low_int]
            if frac > 0.0:
                ret.idxs[idx] = i + wide_range_start
                ret.values[idx] = frac
                idx += 1
                frac_sum += frac
    else:  # One to Many
        # Count how large the arrays have to be to hold the results.
        element_count = wide_range_end - wide_range_start + 1

        ret = RangeSeries(element_count)

        frac = 1.0 / wide_span
        idx = 0
        for i in range(int(wide_range_start), int(wide_range_end + 1)):
            ret.idxs[idx] = i
            ret.values[idx] = frac
            frac_sum += frac
            idx += 1

    # Normalize so sum is 1.0
    for i in range(ret.length):
        ret.values[i] /= frac_sum

    # Weights of ret will always sum to 1.0 (They are normalized)
    return ret


def compute_twit_single_dimension(
        src_start: int, src_end: int, dst_start: int, dst_end: int, w_start: float, w_end: float) -> TwitSingleAxis:
    # print("TWITC _compute_twit_single_dimension(ranges: src %lld, %lld, dst %lld,
    # %lld, weight %f %f\n", src_start, src_end, dst_start, dst_end, w_start,
    # w_end)
    # The + 1 here is because our range is inclusive.
    input_span = abs(src_start - src_end) + 1
    output_span = abs(dst_start - dst_end) + 1
    if input_span == output_span:
        fan = TWIT_FAN_SAME
    elif input_span > output_span:
        fan = TWIT_FAN_IN
    else:
        fan = TWIT_FAN_OUT

    src_inc = twit_sign(src_end - src_start)
    dst_inc = twit_sign(dst_end - dst_start)

    # So....  The smallest length result set for a single axis twit iterator is
    # the larger of the src or dst spans.  The largest
    # is 2x that and the worst case.  Even so this is not large if we over
    # allocate some.  A 200 x 200 x 3 image
    # or neural Concept Map is 120000 values of float about 500k bytes.  The worst
    # case twit
    # single axis iterators will then be 2 x (200 + _ 200 + 3) x 8 indixies and
    # weights are(200 + 200 + 3) x 8 os about
    # 48k bytes.
    sz = max(input_span, output_span)
    if input_span != output_span and input_span != 1 and output_span != 1:
        sz *= 2
    ret = TwitSingleAxis(sz)

    # Now generate the values and normalize
    if fan == TWIT_FAN_SAME:
        dsti = dst_start
        sz = 0
        for srci in range(src_start, src_end + src_inc, src_inc):
            ret.srcidxs[sz] = srci
            ret.dstidxs[sz] = dsti
            ret.weights[sz] = twit_interp(src_start, src_end, w_start, w_end, srci)
            sz += 1
            dsti += dst_inc
    # Normalization not needed for one to one.
    elif fan == TWIT_FAN_IN:
        srci = src_start
        sz = 0
        # Sums for normalizing, by index in span
        sums = np.zeros(output_span, dtype=float)
        for dsti in range(dst_start, dst_end + dst_inc, dst_inc):
            srci += src_inc
            splits = find_range_series_multipliers(dst_start, dst_end, src_start, src_end, dsti)
            for spi in range(0, splits.length):
                ret.srcidxs[sz] = splits.idxs[spi]
                ret.dstidxs[sz] = dsti
                ret.weights[sz] = splits.values[spi]
                sums[(dsti - dst_start) * dst_inc] += ret.weights[sz]
                sz += 1

        # Normalize so each each source sums to 1.0 and also do the interpolation of
        # weights across the span.
        for i in range(0, sz):
            k = ret.dstidxs[i]
            ret.weights[i] *= twit_interp(src_start, src_end, w_start, w_end, ret.srcidxs[i]) / sums[
                (k - dst_start) * dst_inc]
    else:
        # Fan out
        dsti = dst_start
        sz = 0
        # Sums for normalizing, by index in span
        sums = np.zeros(output_span)

        for srci in range(src_start, src_end + src_inc, src_inc):
            dsti += dst_inc
            splits = find_range_series_multipliers(src_start, src_end, dst_start, dst_end, srci)
            for spi in range(0, splits.length):
                ret.srcidxs[sz] = srci
                k = ret.dstidxs[sz] = int(splits.idxs[spi])
                ret.weights[sz] = splits.values[spi]
                sums[k * dst_inc - dst_start] += ret.weights[sz]
                sz += 1

        # Normalize so each each destination sums to 1.0 and also multiply in the
        # interpolated weights across the weight range.
        for i in range(0, sz):
            k = ret.dstidxs[i]
            ret.weights[i] *= twit_interp(dst_start, dst_end, w_start, w_end, ret.dstidxs[i]) / sums[
                k * dst_inc - dst_start]

    if len(ret.weights) > sz:
        ww = np.zeros(sz, dtype=np.float)
        ss = np.zeros(sz, dtype=np.int64)
        dd = np.zeros(sz, dtype=np.int64)
        for i in range(sz):
            ww[i] = ret.weights[i]
            ss[i] = ret.srcidxs[i]
            dd[i] = ret.dstidxs[i]
        ret.length = sz
        ret.weights = ww
        ret.srcidxs = ss
        ret.dstidxs = dd

    # print_twit_single_axis(ret, 0, "TWIT Single Ax: ")
    return ret


# n_dims is the number of dimensions in the source and destination arrays.
# They have to be the same count of dimensions each.
# Possibly some higher dimensions are size 1.
# twit_i is the integer start and ends of ranges in quads, t1_start, t1_end,
# t2_start, t2_end and then repeating for each dimension
# in python order.  twit_w is pairs for dimension weights, w_start, w_end
# repeating in python order for the dimensions.
def compute_twit_multi_dimension(n_dims: int, twit_i: Sequence[int], twit_w: Sequence[float]) -> TwitMultiAxis:
    # Generate axis series
    twit = TwitMultiAxis(n_dims)
    # print("_compute_twit_multi_dimension: n_dims %llD", n_dims)

    dst_start = n_dims * 2
    for i in range(0, n_dims):
        # print(" _compute_twit_multi_dimension: %llD", i)
        wi = srci = i * 2
        dsti = dst_start + i * 2
        twit.axs.append(
            compute_twit_single_dimension(twit_i[srci], twit_i[srci + 1], twit_i[dsti], twit_i[dsti + 1], twit_w[wi],
                                          twit_w[wi + 1]))

    return twit


def apply_twit(twit, t1: np.ndarray, t2: np.ndarray, preclear: int):
    dbg = False
    if dbg:
        print("TWIT  _apply_twit  ")
    if twit is None:
        raise TwitException("apply_twit: twit is None")
    # Src
    if t1 is None:
        raise TwitException("apply_twit: t1 is None")
    # Dst
    if t2 is None:
        raise TwitException("apply_twit: t2 is None")

    t1flat = t1.ravel()
    t2flat = t2.ravel()

    # Fast constants.  This entire method tries top save every cpu cycle possible.
    # Premature optimization is the root of all evil, yada yada yada.
    twit_len = twit.length
    if dbg:
        print("L = %d" % twit_len)

    if twit_len <= 0:
        raise TwitException("twit length <= 0")

    twit_len0 = twit.axs[0].length
    # These three are the source indicies, dset indicies, and weight triples along
    # a given axis.
    # Generated by compute_twit_single_dimension()
    srcidxs0 = twit.axs[0].srcidxs
    dstidxs0 = twit.axs[0].dstidxs
    ws0 = twit.axs[0].weights

    if twit_len == 1:
        if dbg:
            print("_apply_twit  1D")
        if preclear:
            # TODO This preclear may set pixels to 0.0 more than once.
            # Could have a more efficient version that uses the dimension span
            # and directly sets the values, not from srcidxs0[N]
            for i0 in range(0, twit_len0):
                t2flat[dstidxs0[i0]] = 0.0
        if dbg:
            print("Update src")
        for i0 in range(0, twit_len0):
            t2flat[dstidxs0[i0]] += t1flat[srcidxs0[i0]] * ws0[i0]
        return
    else:  # L >= 2
        srcidxs1 = twit.axs[1].srcidxs
        dstidxs1 = twit.axs[1].dstidxs
        ws1 = twit.axs[1].weights
        twit_len1 = twit.axs[1].length

        if twit_len == 2:
            if dbg:
                print("_apply_twit  2D")
            # This is how far a single increment in the next higher axis advances along
            # the source
            # or destination ndarrays.
            srcadvance0 = t1.shape[1]
            dstadvance0 = t2.shape[1]

            # Note: Dimensions are innermost last in the lists!
            # So dim[0] (first dim) changes the slowest and dim[L - 1] (last dim)
            # changes the fastest.

            if preclear:
                for i0 in range(twit_len0):
                    doff0 = dstadvance0 * dstidxs0[i0]
                    for i1 in range(twit_len1):
                        t2flat[dstidxs1[i1] + doff0] = 0.0
            for i0 in range(twit_len0):
                soff0 = srcadvance0 * srcidxs0[i0]
                doff0 = dstadvance0 * dstidxs0[i0]
                w0 = ws0[i0]
                for i1 in range(twit_len1):
                    t2flat[dstidxs1[i1] + doff0] += t1flat[srcidxs1[i1] + soff0] * w0 * ws1[i1]
            return
        else:
            srcidxs2 = twit.axs[2].srcidxs
            dstidxs2 = twit.axs[2].dstidxs
            ws2 = twit.axs[2].weights
            twit_len2 = twit.axs[2].length

            if twit_len == 3:
                if dbg:
                    print("_apply_twit  3D")
                srcadvance1 = t1.shape[2]
                dstadvance1 = t2.shape[2]
                srcadvance0 = t1.shape[1] * srcadvance1
                dstadvance0 = t2.shape[1] * dstadvance1
                if preclear:
                    if dbg:
                        print("  preclear")
                    for i0 in range(twit_len0):
                        doff0 = dstadvance0 * dstidxs0[i0]
                        for i1 in range(twit_len1):
                            doff1 = doff0 + dstadvance1 * dstidxs1[i1]
                            for i2 in range(twit_len2):
                                t2flat[dstidxs2[i2] + doff1] = 0.0
                for i0 in range(twit_len0):
                    if dbg:
                        print("  i0 %d" % i0)
                    soff0 = srcadvance0 * srcidxs0[i0]
                    doff0 = dstadvance0 * dstidxs0[i0]
                    w0 = ws0[i0]
                    for i1 in range(twit_len1):
                        soff1 = soff0 + srcadvance1 * srcidxs1[i1]
                        doff1 = doff0 + dstadvance1 * dstidxs1[i1]
                        w1 = ws1[i1] * w0
                        for i2 in range(twit_len2):
                            # if i2 == 0 or i2 == L2 - 1:
                            # printf(" i2 %llD", i2)
                            # printf("L %lld, %lld %lld i %lld %lld %llD", L0, L1, L2, i0, i1, i2)
                            t2flat[dstidxs2[i2] + doff1] += t1flat[srcidxs2[i2] + soff1] * w1 * ws2[i2]
                if dbg:
                    print("  return ==========================================\n")
                return
            else:
                srcidxs3 = twit.axs[3].srcidxs
                dstidxs3 = twit.axs[3].dstidxs
                ws3 = twit.axs[3].weights
                twit_len3 = twit.axs[3].length
                if twit_len == 4:
                    if dbg:
                        print("_apply_twit  4D")
                    srcadvance2 = t1.shape[3]
                    dstadvance2 = t2.shape[3]
                    srcadvance1 = t1.shape[2] * srcadvance2
                    dstadvance1 = t2.shape[2] * dstadvance2
                    srcadvance0 = t1.shape[1] * srcadvance1
                    dstadvance0 = t2.shape[1] * dstadvance1
                    if preclear:
                        for i0 in range(twit_len0):
                            doff0 = dstadvance0 * dstidxs0[i0]
                            for i1 in range(twit_len1):
                                doff1 = doff0 + dstadvance1 * dstidxs1[i1]
                                for i2 in range(twit_len2):
                                    doff2 = doff1 + dstadvance2 * dstidxs2[i2]
                                    for i3 in range(twit_len3):
                                        t2flat[dstidxs3[i3] + doff2] = 0.0
                    for i0 in range(twit_len0):
                        soff0 = srcadvance0 * srcidxs0[i0]
                        doff0 = dstadvance0 * dstidxs0[i0]
                        w0 = ws0[i0]
                        for i1 in range(twit_len1):
                            soff1 = soff0 + srcadvance1 * srcidxs1[i1]
                            doff1 = doff0 + dstadvance1 * dstidxs1[i1]
                            w1 = ws1[i1] * w0
                            for i2 in range(twit_len2):
                                soff2 = soff1 + srcadvance2 * srcidxs2[i2]
                                doff2 = doff1 + dstadvance2 * dstidxs2[i2]
                                w2 = ws2[i2] * w1
                                for i3 in range(twit_len3):
                                    t2flat[dstidxs3[i3] + doff2] += t1flat[srcidxs3[i3] + soff2] * w2 * ws3[i3]
                    return
                else:
                    srcidxs4 = twit.axs[4].srcidxs
                    dstidxs4 = twit.axs[4].dstidxs
                    ws4 = twit.axs[4].weights
                    twit_len4 = twit.axs[4].length
                    if twit_len == 5:
                        if dbg:
                            print("_apply_twit  5D")
                        srcadvance3 = t1.shape[4]
                        dstadvance3 = t2.shape[4]
                        srcadvance2 = t1.shape[3] * srcadvance3
                        dstadvance2 = t2.shape[3] * dstadvance3
                        srcadvance1 = t1.shape[2] * srcadvance2
                        dstadvance1 = t2.shape[2] * dstadvance2
                        srcadvance0 = t1.shape[1] * srcadvance1
                        dstadvance0 = t2.shape[1] * dstadvance1
                        if preclear:
                            for i0 in range(twit_len0):
                                doff0 = dstadvance0 * dstidxs0[i0]
                                for i1 in range(twit_len1):
                                    doff1 = doff0 + dstadvance1 * dstidxs1[i1]
                                    for i2 in range(twit_len2):
                                        doff2 = doff1 + dstadvance2 * dstidxs2[i2]
                                        for i3 in range(twit_len3):
                                            doff3 = doff2 + dstadvance3 * dstidxs3[i3]
                                            for i4 in range(twit_len4):
                                                t2flat[dstidxs4[i4] + doff3] = 0.0
                        for i0 in range(twit_len0):
                            soff0 = srcadvance0 * srcidxs0[i0]
                            doff0 = dstadvance0 * dstidxs0[i0]
                            w0 = ws0[i0]
                            for i1 in range(twit_len1):
                                soff1 = soff0 + srcadvance1 * srcidxs1[i1]
                                doff1 = doff0 + dstadvance1 * dstidxs1[i1]
                                w1 = ws1[i1] * w0
                                for i2 in range(twit_len2):
                                    soff2 = soff1 + srcadvance2 * srcidxs2[i2]
                                    doff2 = doff1 + dstadvance2 * dstidxs2[i2]
                                    w2 = ws2[i2] * w1
                                    for i3 in range(twit_len3):
                                        soff3 = soff2 + srcadvance3 * srcidxs3[i3]
                                        doff3 = doff2 + dstadvance3 * dstidxs3[i3]
                                        w3 = ws3[i3] * w2
                                        for i4 in range(twit_len4):
                                            t2flat[dstidxs4[i4] + doff3] += t1flat[srcidxs4[i4] + soff3] * w3 * ws4[i4]
                        return
                    else:
                        srcidxs5 = twit.axs[5].srcidxs
                        dstidxs5 = twit.axs[5].dstidxs
                        ws5 = twit.axs[5].weights
                        twit_len5 = twit.axs[5].length
                        if twit_len == 6:
                            if dbg:
                                print("_apply_twit  6D")
                            srcadvance4 = t1.shape[5]
                            dstadvance4 = t2.shape[5]
                            srcadvance3 = t1.shape[4] * srcadvance4
                            dstadvance3 = t2.shape[4] * dstadvance4
                            srcadvance2 = t1.shape[3] * srcadvance3
                            dstadvance2 = t2.shape[3] * dstadvance3
                            srcadvance1 = t1.shape[2] * srcadvance2
                            dstadvance1 = t2.shape[2] * dstadvance2
                            srcadvance0 = t1.shape[1] * srcadvance1
                            dstadvance0 = t2.shape[1] * dstadvance1
                            if preclear:
                                for i0 in range(twit_len0):
                                    doff0 = dstadvance0 * dstidxs0[i0]
                                    for i1 in range(twit_len1):
                                        doff1 = doff0 + dstadvance1 * dstidxs1[i1]
                                        for i2 in range(twit_len2):
                                            doff2 = doff1 + dstadvance2 * dstidxs2[i2]
                                            for i3 in range(twit_len3):
                                                doff3 = doff2 + dstadvance3 * dstidxs3[i3]
                                                for i4 in range(twit_len4):
                                                    doff4 = doff3 + dstadvance4 * dstidxs4[i4]
                                                    for i5 in range(twit_len5):
                                                        t2flat[dstidxs5[i5] + doff4] = 0.0
                            for i0 in range(twit_len0):
                                soff0 = srcadvance0 * srcidxs0[i0]
                                doff0 = dstadvance0 * dstidxs0[i0]
                                w0 = ws0[i0]
                                for i1 in range(twit_len1):
                                    soff1 = soff0 + srcadvance1 * srcidxs1[i1]
                                    doff1 = doff0 + dstadvance1 * dstidxs1[i1]
                                    w1 = ws1[i1] * w0
                                    for i2 in range(twit_len2):
                                        soff2 = soff1 + srcadvance2 * srcidxs2[i2]
                                        doff2 = doff1 + dstadvance2 * dstidxs2[i2]
                                        w2 = ws2[i2] * w1
                                        for i3 in range(twit_len3):
                                            soff3 = soff2 + srcadvance3 * srcidxs3[i3]
                                            doff3 = doff2 + dstadvance3 * dstidxs3[i3]
                                            w3 = ws3[i3] * w2
                                            for i4 in range(twit_len4):
                                                soff4 = soff3 + srcadvance4 * srcidxs4[i4]
                                                doff4 = doff3 + dstadvance4 * dstidxs4[i4]
                                                w4 = ws4[i4] * w3
                                                for i5 in range(twit_len5):
                                                    t2flat[dstidxs5[i5] + doff4] += t1flat[srcidxs5[i5] + soff4] * \
                                                                                    w4 * ws5[i5]
                            return
                        else:
                            # Tsk tsk tsk, unimplemented number of dimensions.
                            # They're all custom for each supported dimension count.
                            # May implement a slower generic size handler for large numbers of
                            # dimensions?
                            print("_apply_twit  UNSUPPORTED TWIT Dimensions count.  Max is 6 Dimensions.")
                            raise TwitException("_apply_twit  UNSUPPORTED TWIT Dimensions count.  Max is 6 Dimensions.")


if __name__ == '__main__':
    pass
