static int hdcd_scan(HDCDContext *ctx, hdcd_state_t *state, const int32_t *samples, int max, int stride)

{

    int cdt_active = 0;

    /* code detect timer */

    int result;

    if (state->sustain > 0) {

        cdt_active = 1;

        if (state->sustain <= max) {

            state->control = 0;

            max = state->sustain;

        }

        state->sustain -= max;

    }

    result = 0;

    while (result < max) {

        int flag;

        int consumed = hdcd_integrate(ctx, state, &flag, samples, max - result, stride);

        result += consumed;

        if (flag > 0) {

            /* reset timer if code detected in channel */

            hdcd_sustain_reset(state);

            break;

        }

        samples += consumed * stride;

    }

    /* code detect timer expired */

    if (cdt_active && state->sustain == 0)

        state->count_sustain_expired++;

    return result;

}
