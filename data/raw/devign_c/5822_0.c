static inline void set_fsr(CPUSPARCState *env)

{

    int rnd_mode;



    switch (env->fsr & FSR_RD_MASK) {

    case FSR_RD_NEAREST:

        rnd_mode = float_round_nearest_even;

        break;

    default:

    case FSR_RD_ZERO:

        rnd_mode = float_round_to_zero;

        break;

    case FSR_RD_POS:

        rnd_mode = float_round_up;

        break;

    case FSR_RD_NEG:

        rnd_mode = float_round_down;

        break;

    }

    set_float_rounding_mode(rnd_mode, &env->fp_status);

}
