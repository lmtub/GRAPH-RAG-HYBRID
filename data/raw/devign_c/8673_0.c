static int64_t do_strtosz(const char *nptr, char **end,

                          const char default_suffix, int64_t unit)

{

    int64_t retval;

    char *endptr;

    unsigned char c;

    int mul_required = 0;

    double val, mul, integral, fraction;



    errno = 0;

    val = strtod(nptr, &endptr);

    if (isnan(val) || endptr == nptr || errno != 0) {

        retval = -EINVAL;

        goto out;

    }

    fraction = modf(val, &integral);

    if (fraction != 0) {

        mul_required = 1;

    }

    c = *endptr;

    mul = suffix_mul(c, unit);

    if (mul >= 0) {

        endptr++;

    } else {

        mul = suffix_mul(default_suffix, unit);

        assert(mul >= 0);

    }

    if (mul == 1 && mul_required) {

        retval = -EINVAL;

        goto out;

    }

    if ((val * mul >= INT64_MAX) || val < 0) {

        retval = -ERANGE;

        goto out;

    }

    retval = val * mul;



out:

    if (end) {

        *end = endptr;

    } else if (*endptr) {

        retval = -EINVAL;

    }



    return retval;

}
