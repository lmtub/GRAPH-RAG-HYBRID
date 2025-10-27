static int revert_channel_correlation(ALSDecContext *ctx, ALSBlockData *bd,

                                       ALSChannelData **cd, int *reverted,

                                       unsigned int offset, int c)

{

    ALSChannelData *ch = cd[c];

    unsigned int   dep = 0;

    unsigned int channels = ctx->avctx->channels;



    if (reverted[c])

        return 0;



    reverted[c] = 1;



    while (dep < channels && !ch[dep].stop_flag) {

        revert_channel_correlation(ctx, bd, cd, reverted, offset,

                                   ch[dep].master_channel);



        dep++;

    }



    if (dep == channels) {

        av_log(ctx->avctx, AV_LOG_WARNING, "Invalid channel correlation.\n");

        return AVERROR_INVALIDDATA;

    }



    bd->const_block = ctx->const_block + c;

    bd->shift_lsbs  = ctx->shift_lsbs + c;

    bd->opt_order   = ctx->opt_order + c;

    bd->store_prev_samples = ctx->store_prev_samples + c;

    bd->use_ltp     = ctx->use_ltp + c;

    bd->ltp_lag     = ctx->ltp_lag + c;

    bd->ltp_gain    = ctx->ltp_gain[c];

    bd->lpc_cof     = ctx->lpc_cof[c];

    bd->quant_cof   = ctx->quant_cof[c];

    bd->raw_samples = ctx->raw_samples[c] + offset;



    dep = 0;

    while (!ch[dep].stop_flag) {

        unsigned int smp;

        unsigned int begin = 1;

        unsigned int end   = bd->block_length - 1;

        int64_t y;

        int32_t *master = ctx->raw_samples[ch[dep].master_channel] + offset;



        if (ch[dep].time_diff_flag) {

            int t = ch[dep].time_diff_index;



            if (ch[dep].time_diff_sign) {

                t      = -t;

                begin -= t;

            } else {

                end   -= t;

            }



            for (smp = begin; smp < end; smp++) {

                y  = (1 << 6) +

                     MUL64(ch[dep].weighting[0], master[smp - 1    ]) +

                     MUL64(ch[dep].weighting[1], master[smp        ]) +

                     MUL64(ch[dep].weighting[2], master[smp + 1    ]) +

                     MUL64(ch[dep].weighting[3], master[smp - 1 + t]) +

                     MUL64(ch[dep].weighting[4], master[smp     + t]) +

                     MUL64(ch[dep].weighting[5], master[smp + 1 + t]);



                bd->raw_samples[smp] += y >> 7;

            }

        } else {

            for (smp = begin; smp < end; smp++) {

                y  = (1 << 6) +

                     MUL64(ch[dep].weighting[0], master[smp - 1]) +

                     MUL64(ch[dep].weighting[1], master[smp    ]) +

                     MUL64(ch[dep].weighting[2], master[smp + 1]);



                bd->raw_samples[smp] += y >> 7;

            }

        }



        dep++;

    }



    return 0;

}
