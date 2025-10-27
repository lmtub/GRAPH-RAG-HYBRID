static void lowpass_line_complex_c(uint8_t *dstp, ptrdiff_t linesize,

                                   const uint8_t *srcp,

                                   ptrdiff_t mref, ptrdiff_t pref)

{

    const uint8_t *srcp_above = srcp + mref;

    const uint8_t *srcp_below = srcp + pref;

    const uint8_t *srcp_above2 = srcp + mref * 2;

    const uint8_t *srcp_below2 = srcp + pref * 2;

    int i;

    for (i = 0; i < linesize; i++) {

        // this calculation is an integer representation of

        // '0.75 * current + 0.25 * above + 0.25 * below - 0.125 * above2 - 0.125 * below2'

        // '4 +' is for rounding.

        dstp[i] = av_clip_uint8((4 + (srcp[i] << 2)

                  + ((srcp[i] + srcp_above[i] + srcp_below[i]) << 1)

                  - srcp_above2[i] - srcp_below2[i]) >> 3);

    }

}
