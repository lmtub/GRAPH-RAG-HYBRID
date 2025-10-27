float64 HELPER(recpe_f64)(float64 input, void *fpstp)

{

    float_status *fpst = fpstp;

    float64 f64 = float64_squash_input_denormal(input, fpst);

    uint64_t f64_val = float64_val(f64);

    uint64_t f64_sbit = 0x8000000000000000ULL & f64_val;

    int64_t f64_exp = extract64(f64_val, 52, 11);

    float64 r64;

    uint64_t r64_val;

    int64_t r64_exp;

    uint64_t r64_frac;



    /* Deal with any special cases */

    if (float64_is_any_nan(f64)) {

        float64 nan = f64;

        if (float64_is_signaling_nan(f64)) {

            float_raise(float_flag_invalid, fpst);

            nan = float64_maybe_silence_nan(f64);

        }

        if (fpst->default_nan_mode) {

            nan =  float64_default_nan;

        }

        return nan;

    } else if (float64_is_infinity(f64)) {

        return float64_set_sign(float64_zero, float64_is_neg(f64));

    } else if (float64_is_zero(f64)) {

        float_raise(float_flag_divbyzero, fpst);

        return float64_set_sign(float64_infinity, float64_is_neg(f64));

    } else if ((f64_val & ~(1ULL << 63)) < (1ULL << 50)) {

        /* Abs(value) < 2.0^-1024 */

        float_raise(float_flag_overflow | float_flag_inexact, fpst);

        if (round_to_inf(fpst, f64_sbit)) {

            return float64_set_sign(float64_infinity, float64_is_neg(f64));

        } else {

            return float64_set_sign(float64_maxnorm, float64_is_neg(f64));

        }

    } else if (f64_exp >= 1023 && fpst->flush_to_zero) {

        float_raise(float_flag_underflow, fpst);

        return float64_set_sign(float64_zero, float64_is_neg(f64));

    }



    r64 = call_recip_estimate(f64, 2045, fpst);

    r64_val = float64_val(r64);

    r64_exp = extract64(r64_val, 52, 11);

    r64_frac = extract64(r64_val, 0, 52);



    /* result = sign : result_exp<10:0> : fraction<51:0> */

    return make_float64(f64_sbit |

                        ((r64_exp & 0x7ff) << 52) |

                        r64_frac);

}
