uint32_t helper_fcmp_un(uint32_t a, uint32_t b)

{

    CPU_FloatU fa, fb;

    uint32_t r = 0;



    fa.l = a;

    fb.l = b;



    if (float32_is_signaling_nan(fa.f) || float32_is_signaling_nan(fb.f)) {

        update_fpu_flags(float_flag_invalid);

        r = 1;

    }



    if (float32_is_nan(fa.f) || float32_is_nan(fb.f)) {

        r = 1;

    }



    return r;

}
