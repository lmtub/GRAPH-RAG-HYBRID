static int decode_blocks(ALSDecContext *ctx, unsigned int ra_frame,

                         unsigned int c, const unsigned int *div_blocks,

                         unsigned int *js_blocks)

{

    ALSSpecificConfig *sconf = &ctx->sconf;

    unsigned int offset = 0;

    unsigned int b;

    ALSBlockData bd[2] = { { 0 } };



    bd[0].ra_block         = ra_frame;

    bd[0].const_block      = ctx->const_block;

    bd[0].shift_lsbs       = ctx->shift_lsbs;

    bd[0].opt_order        = ctx->opt_order;

    bd[0].store_prev_samples = ctx->store_prev_samples;

    bd[0].use_ltp          = ctx->use_ltp;

    bd[0].ltp_lag          = ctx->ltp_lag;

    bd[0].ltp_gain         = ctx->ltp_gain[0];

    bd[0].quant_cof        = ctx->quant_cof[0];

    bd[0].lpc_cof          = ctx->lpc_cof[0];

    bd[0].prev_raw_samples = ctx->prev_raw_samples;

    bd[0].js_blocks        = *js_blocks;



    bd[1].ra_block         = ra_frame;

    bd[1].const_block      = ctx->const_block;

    bd[1].shift_lsbs       = ctx->shift_lsbs;

    bd[1].opt_order        = ctx->opt_order;

    bd[1].store_prev_samples = ctx->store_prev_samples;

    bd[1].use_ltp          = ctx->use_ltp;

    bd[1].ltp_lag          = ctx->ltp_lag;

    bd[1].ltp_gain         = ctx->ltp_gain[0];

    bd[1].quant_cof        = ctx->quant_cof[0];

    bd[1].lpc_cof          = ctx->lpc_cof[0];

    bd[1].prev_raw_samples = ctx->prev_raw_samples;

    bd[1].js_blocks        = *(js_blocks + 1);



    // decode all blocks

    for (b = 0; b < ctx->num_blocks; b++) {

        unsigned int s;



        bd[0].block_length = div_blocks[b];

        bd[1].block_length = div_blocks[b];



        bd[0].raw_samples  = ctx->raw_samples[c    ] + offset;

        bd[1].raw_samples  = ctx->raw_samples[c + 1] + offset;



        bd[0].raw_other    = bd[1].raw_samples;

        bd[1].raw_other    = bd[0].raw_samples;



        if(read_decode_block(ctx, &bd[0]) || read_decode_block(ctx, &bd[1])) {

            // damaged block, write zero for the rest of the frame

            zero_remaining(b, ctx->num_blocks, div_blocks, bd[0].raw_samples);

            zero_remaining(b, ctx->num_blocks, div_blocks, bd[1].raw_samples);

            return -1;

        }



        // reconstruct joint-stereo blocks

        if (bd[0].js_blocks) {

            if (bd[1].js_blocks)

                av_log(ctx->avctx, AV_LOG_WARNING, "Invalid channel pair!\n");



            for (s = 0; s < div_blocks[b]; s++)

                bd[0].raw_samples[s] = bd[1].raw_samples[s] - bd[0].raw_samples[s];

        } else if (bd[1].js_blocks) {

            for (s = 0; s < div_blocks[b]; s++)

                bd[1].raw_samples[s] = bd[1].raw_samples[s] + bd[0].raw_samples[s];

        }



        offset  += div_blocks[b];

        bd[0].ra_block = 0;

        bd[1].ra_block = 0;

    }



    // store carryover raw samples,

    // the others channel raw samples are stored by the calling function.

    memmove(ctx->raw_samples[c] - sconf->max_order,

            ctx->raw_samples[c] - sconf->max_order + sconf->frame_length,

            sizeof(*ctx->raw_samples[c]) * sconf->max_order);



    return 0;

}
