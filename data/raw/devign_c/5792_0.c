static int hdcd_integrate(HDCDContext *ctx, hdcd_state_t *state, int *flag, const int32_t *samples, int count, int stride)

{

    uint32_t bits = 0;

    int result = FFMIN(state->readahead, count);

    int i;

    *flag = 0;



    for (i = result - 1; i >= 0; i--) {

        bits |= (*samples & 1) << i; /* might be better as a conditional? */

        samples += stride;

    }



    state->window = (state->window << result) | bits;

    state->readahead -= result;

    if (state->readahead > 0)

        return result;



    bits = (state->window ^ state->window >> 5 ^ state->window >> 23);



    if (state->arg) {

        if ((bits & 0x0fa00500) == 0x0fa00500) {

            /* A: 8-bit code */

            if ((bits & 0xc8) == 0) {

                /*                   [..pt gggg]

                 * 0x0fa005[..] -> 0b[00.. 0...], gain part doubled */

                state->control = (bits & 255) + (bits & 7);

                *flag = 1;

                state->code_counterA++;

            } else {

                /* one of bits 3, 6, or 7 was not 0 */

                state->code_counterA_almost++;

                av_log(ctx->fctx, AV_LOG_VERBOSE,

                    "hdcd error: Control A almost: 0x%02x near %d\n", bits & 0xff, ctx->sample_count);

            }

        } else if ((bits & 0xa0060000) == 0xa0060000) {

            /* B: 8-bit code, 8-bit XOR check */

            if (((bits ^ (~bits >> 8 & 255)) & 0xffff00ff) == 0xa0060000) {

                /*          check:   [..pt gggg ~(..pt gggg)]

                 * 0xa006[....] -> 0b[.... ....   .... .... ] */

                state->control = bits >> 8 & 255;

                *flag = 1;

                state->code_counterB++;

            } else {

                /* XOR check failed */

                state->code_counterB_checkfails++;

                av_log(ctx->fctx, AV_LOG_VERBOSE,

                       "hdcd error: Control B check failed: 0x%04x (0x%02x vs 0x%02x) near %d\n", bits & 0xffff, (bits & 0xff00) >> 8, ~bits & 0xff, ctx->sample_count);

            }

        } else {

            /* told to look for a code, but didn't match one */

            state->code_counterC_unmatched++;

            av_log(ctx->fctx, AV_LOG_VERBOSE,

                   "hdcd error: Unmatched code: 0x%08x near %d\n", bits, ctx->sample_count);

        }

        if (*flag) hdcd_update_info(state);

        state->arg = 0;

    }

    if (bits == 0x7e0fa005 || bits == 0x7e0fa006) {

        state->readahead = (bits & 3) * 8;

        state->arg = 1;

        state->code_counterC++;

    } else {

        if (bits)

            state->readahead = readaheadtab[bits & 0xff];

        else

            state->readahead = 31; /* ffwd over digisilence */

    }

    return result;

}
