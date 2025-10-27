static void write_header(FFV1Context *f)

{

    uint8_t state[CONTEXT_SIZE];

    int i, j;

    RangeCoder *const c = &f->slice_context[0]->c;



    memset(state, 128, sizeof(state));



    if (f->version < 2) {

        put_symbol(c, state, f->version, 0);

        put_symbol(c, state, f->ac, 0);

        if (f->ac > 1) {

            for (i = 1; i < 256; i++)

                put_symbol(c, state,

                           f->state_transition[i] - c->one_state[i], 1);

        }

        put_symbol(c, state, f->colorspace, 0); // YUV cs type

        if (f->version > 0)

            put_symbol(c, state, f->bits_per_raw_sample, 0);

        put_rac(c, state, f->chroma_planes);

        put_symbol(c, state, f->chroma_h_shift, 0);

        put_symbol(c, state, f->chroma_v_shift, 0);

        put_rac(c, state, f->transparency);



        write_quant_tables(c, f->quant_table);

    } else if (f->version < 3) {

        put_symbol(c, state, f->slice_count, 0);

        for (i = 0; i < f->slice_count; i++) {

            FFV1Context *fs = f->slice_context[i];

            put_symbol(c, state,

                       (fs->slice_x      + 1) * f->num_h_slices / f->width, 0);

            put_symbol(c, state,

                       (fs->slice_y      + 1) * f->num_v_slices / f->height, 0);

            put_symbol(c, state,

                       (fs->slice_width  + 1) * f->num_h_slices / f->width - 1,

                       0);

            put_symbol(c, state,

                       (fs->slice_height + 1) * f->num_v_slices / f->height - 1,

                       0);

            for (j = 0; j < f->plane_count; j++) {

                put_symbol(c, state, f->plane[j].quant_table_index, 0);

                av_assert0(f->plane[j].quant_table_index == f->avctx->context_model);

            }

        }

    }

}
