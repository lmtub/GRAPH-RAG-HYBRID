static int calc_bit_demand(AacPsyContext *ctx, float pe, int bits, int size,

                           int short_window)

{

    const float bitsave_slope  = short_window ? PSY_3GPP_SAVE_SLOPE_S  : PSY_3GPP_SAVE_SLOPE_L;

    const float bitsave_add    = short_window ? PSY_3GPP_SAVE_ADD_S    : PSY_3GPP_SAVE_ADD_L;

    const float bitspend_slope = short_window ? PSY_3GPP_SPEND_SLOPE_S : PSY_3GPP_SPEND_SLOPE_L;

    const float bitspend_add   = short_window ? PSY_3GPP_SPEND_ADD_S   : PSY_3GPP_SPEND_ADD_L;

    const float clip_low       = short_window ? PSY_3GPP_CLIP_LO_S     : PSY_3GPP_CLIP_LO_L;

    const float clip_high      = short_window ? PSY_3GPP_CLIP_HI_S     : PSY_3GPP_CLIP_HI_L;

    float clipped_pe, bit_save, bit_spend, bit_factor, fill_level;



    ctx->fill_level += ctx->frame_bits - bits;

    ctx->fill_level  = av_clip(ctx->fill_level, 0, size);

    fill_level = av_clipf((float)ctx->fill_level / size, clip_low, clip_high);

    clipped_pe = av_clipf(pe, ctx->pe.min, ctx->pe.max);

    bit_save   = (fill_level + bitsave_add) * bitsave_slope;

    assert(bit_save <= 0.3f && bit_save >= -0.05000001f);

    bit_spend  = (fill_level + bitspend_add) * bitspend_slope;

    assert(bit_spend <= 0.5f && bit_spend >= -0.1f);

    /* The bit factor graph in the spec is obviously incorrect.

     *      bit_spend + ((bit_spend - bit_spend))...

     * The reference encoder subtracts everything from 1, but also seems incorrect.

     *      1 - bit_save + ((bit_spend + bit_save))...

     * Hopefully below is correct.

     */

    bit_factor = 1.0f - bit_save + ((bit_spend - bit_save) / (ctx->pe.max - ctx->pe.min)) * (clipped_pe - ctx->pe.min);

    /* NOTE: The reference encoder attempts to center pe max/min around the current pe. */

    ctx->pe.max = FFMAX(pe, ctx->pe.max);

    ctx->pe.min = FFMIN(pe, ctx->pe.min);



    return FFMIN(ctx->frame_bits * bit_factor, ctx->frame_bits + size - bits);

}
