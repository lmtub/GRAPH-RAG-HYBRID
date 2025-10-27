static int write_extradata(FFV1Context *f)

{

    RangeCoder *const c = &f->c;

    uint8_t state[CONTEXT_SIZE];

    int i, j, k;

    uint8_t state2[32][CONTEXT_SIZE];

    unsigned v;



    memset(state2, 128, sizeof(state2));

    memset(state, 128, sizeof(state));



    f->avctx->extradata_size = 10000 + 4 +

                                    (11 * 11 * 5 * 5 * 5 + 11 * 11 * 11) * 32;

    f->avctx->extradata = av_malloc(f->avctx->extradata_size);



    ff_init_range_encoder(c, f->avctx->extradata, f->avctx->extradata_size);

    ff_build_rac_states(c, 0.05 * (1LL << 32), 256 - 8);



    put_symbol(c, state, f->version, 0);

    if (f->version > 2) {

        if (f->version == 3)

            f->minor_version = 4;

        put_symbol(c, state, f->minor_version, 0);

    }



    put_symbol(c, state, f->ac, 0);

    if (f->ac > 1)

        for (i = 1; i < 256; i++)

            put_symbol(c, state, f->state_transition[i] - c->one_state[i], 1);



    put_symbol(c, state, f->colorspace, 0); // YUV cs type

    put_symbol(c, state, f->bits_per_raw_sample, 0);

    put_rac(c, state, f->chroma_planes);

    put_symbol(c, state, f->chroma_h_shift, 0);

    put_symbol(c, state, f->chroma_v_shift, 0);

    put_rac(c, state, f->transparency);

    put_symbol(c, state, f->num_h_slices - 1, 0);

    put_symbol(c, state, f->num_v_slices - 1, 0);



    put_symbol(c, state, f->quant_table_count, 0);

    for (i = 0; i < f->quant_table_count; i++)

        write_quant_tables(c, f->quant_tables[i]);



    for (i = 0; i < f->quant_table_count; i++) {

        for (j = 0; j < f->context_count[i] * CONTEXT_SIZE; j++)

            if (f->initial_states[i] && f->initial_states[i][0][j] != 128)

                break;

        if (j < f->context_count[i] * CONTEXT_SIZE) {

            put_rac(c, state, 1);

            for (j = 0; j < f->context_count[i]; j++)

                for (k = 0; k < CONTEXT_SIZE; k++) {

                    int pred = j ? f->initial_states[i][j - 1][k] : 128;

                    put_symbol(c, state2[k],

                               (int8_t)(f->initial_states[i][j][k] - pred), 1);

                }

        } else {

            put_rac(c, state, 0);

        }

    }



    if (f->version > 2) {

        put_symbol(c, state, f->ec, 0);

        put_symbol(c, state, f->intra = (f->avctx->gop_size < 2), 0);

    }



    f->avctx->extradata_size = ff_rac_terminate(c);

    v = av_crc(av_crc_get_table(AV_CRC_32_IEEE), 0, f->avctx->extradata, f->avctx->extradata_size);

    AV_WL32(f->avctx->extradata + f->avctx->extradata_size, v);

    f->avctx->extradata_size += 4;



    return 0;

}