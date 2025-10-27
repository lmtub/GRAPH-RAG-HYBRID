static uint64_t do_cvttq(CPUAlphaState *env, uint64_t a, int roundmode)

{

    uint64_t frac, ret = 0;

    uint32_t exp, sign, exc = 0;

    int shift;



    sign = (a >> 63);

    exp = (uint32_t)(a >> 52) & 0x7ff;

    frac = a & 0xfffffffffffffull;



    if (exp == 0) {

        if (unlikely(frac != 0)) {

            goto do_underflow;

        }

    } else if (exp == 0x7ff) {

        exc = FPCR_INV;

    } else {

        /* Restore implicit bit.  */

        frac |= 0x10000000000000ull;



        shift = exp - 1023 - 52;

        if (shift >= 0) {

            /* In this case the number is so large that we must shift

               the fraction left.  There is no rounding to do.  */

            if (shift < 64) {

                ret = frac << shift;

            }

            /* Check for overflow.  Note the special case of -0x1p63.  */

            if (shift >= 11 && a != 0xC3E0000000000000ull) {

                exc = FPCR_IOV | FPCR_INE;

            }

        } else {

            uint64_t round;



            /* In this case the number is smaller than the fraction as

               represented by the 52 bit number.  Here we must think

               about rounding the result.  Handle this by shifting the

               fractional part of the number into the high bits of ROUND.

               This will let us efficiently handle round-to-nearest.  */

            shift = -shift;

            if (shift < 63) {

                ret = frac >> shift;

                round = frac << (64 - shift);

            } else {

                /* The exponent is so small we shift out everything.

                   Leave a sticky bit for proper rounding below.  */

            do_underflow:

                round = 1;

            }



            if (round) {

                exc = FPCR_INE;

                switch (roundmode) {

                case float_round_nearest_even:

                    if (round == (1ull << 63)) {

                        /* Fraction is exactly 0.5; round to even.  */

                        ret += (ret & 1);

                    } else if (round > (1ull << 63)) {

                        ret += 1;

                    }

                    break;

                case float_round_to_zero:

                    break;

                case float_round_up:

                    ret += 1 - sign;

                    break;

                case float_round_down:

                    ret += sign;

                    break;

                }

            }

        }

        if (sign) {

            ret = -ret;

        }

    }

    env->error_code = exc;



    return ret;

}
