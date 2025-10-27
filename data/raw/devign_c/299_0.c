static int ac3_encode_frame(AVCodecContext *avctx,

                            unsigned char *frame, int buf_size, void *data)

{

    AC3EncodeContext *s = avctx->priv_data;

    const int16_t *samples = data;

    int16_t planar_samples[AC3_MAX_CHANNELS][AC3_BLOCK_SIZE+AC3_FRAME_SIZE];

    int32_t mdct_coef[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS][AC3_MAX_COEFS];

    uint8_t exp[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS][AC3_MAX_COEFS];

    uint8_t exp_strategy[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS];

    uint8_t encoded_exp[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS][AC3_MAX_COEFS];

    uint8_t num_exp_groups[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS];

    uint8_t grouped_exp[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS][AC3_MAX_EXP_GROUPS];

    uint8_t bap[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS][AC3_MAX_COEFS];

    int8_t exp_shift[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS];

    uint16_t qmant[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS][AC3_MAX_COEFS];

    int frame_bits;



    if (s->bit_alloc.sr_code == 1)

        adjust_frame_size(s);



    deinterleave_input_samples(s, samples, planar_samples);



    apply_mdct(s, planar_samples, exp_shift, mdct_coef);



    frame_bits = process_exponents(s, mdct_coef, exp_shift, exp, exp_strategy,

                                   encoded_exp, num_exp_groups, grouped_exp);



    compute_bit_allocation(s, bap, encoded_exp, exp_strategy, frame_bits);



    quantize_mantissas(s, mdct_coef, exp_shift, encoded_exp, bap, qmant);



    output_frame(s, frame, exp_strategy, num_exp_groups, grouped_exp, bap, qmant);



    return s->frame_size;

}
