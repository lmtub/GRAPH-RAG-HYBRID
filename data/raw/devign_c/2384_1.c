static void decode_pitch_lag_low(int *lag_int, int *lag_frac, int pitch_index,

                                 uint8_t *base_lag_int, int subframe, enum Mode mode)

{

    if (subframe == 0 || (subframe == 2 && mode != MODE_6k60)) {

        if (pitch_index < 116) {

            *lag_int  = (pitch_index + 69) >> 1;

            *lag_frac = (pitch_index - (*lag_int << 1) + 68) << 1;

        } else {

            *lag_int  = pitch_index - 24;

            *lag_frac = 0;

        }

        // XXX: same problem as before

        *base_lag_int = av_clip(*lag_int - 8 - (*lag_frac < 0),

                                AMRWB_P_DELAY_MIN, AMRWB_P_DELAY_MAX - 15);

    } else {

        *lag_int  = (pitch_index + 1) >> 1;

        *lag_frac = (pitch_index - (*lag_int << 1)) << 1;

        *lag_int += *base_lag_int;

    }

}
