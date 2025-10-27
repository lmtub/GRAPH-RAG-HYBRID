static void vp56_decode_mb(VP56Context *s, int row, int col, int is_alpha)

{

    AVFrame *frame_current, *frame_ref;

    VP56mb mb_type;

    VP56Frame ref_frame;

    int b, ab, b_max, plane, off;



    if (s->framep[VP56_FRAME_CURRENT]->key_frame)

        mb_type = VP56_MB_INTRA;

    else

        mb_type = vp56_decode_mv(s, row, col);

    ref_frame = vp56_reference_frame[mb_type];



    s->dsp.clear_blocks(*s->block_coeff);



    s->parse_coeff(s);



    vp56_add_predictors_dc(s, ref_frame);



    frame_current = s->framep[VP56_FRAME_CURRENT];

    frame_ref = s->framep[ref_frame];

    if (mb_type != VP56_MB_INTRA && !frame_ref->data[0])

        return;



    ab = 6*is_alpha;

    b_max = 6 - 2*is_alpha;



    switch (mb_type) {

        case VP56_MB_INTRA:

            for (b=0; b<b_max; b++) {

                plane = ff_vp56_b2p[b+ab];

                s->dsp.idct_put(frame_current->data[plane] + s->block_offset[b],

                                s->stride[plane], s->block_coeff[b]);

            }

            break;



        case VP56_MB_INTER_NOVEC_PF:

        case VP56_MB_INTER_NOVEC_GF:

            for (b=0; b<b_max; b++) {

                plane = ff_vp56_b2p[b+ab];

                off = s->block_offset[b];

                s->dsp.put_pixels_tab[1][0](frame_current->data[plane] + off,

                                            frame_ref->data[plane] + off,

                                            s->stride[plane], 8);

                s->dsp.idct_add(frame_current->data[plane] + off,

                                s->stride[plane], s->block_coeff[b]);

            }

            break;



        case VP56_MB_INTER_DELTA_PF:

        case VP56_MB_INTER_V1_PF:

        case VP56_MB_INTER_V2_PF:

        case VP56_MB_INTER_DELTA_GF:

        case VP56_MB_INTER_4V:

        case VP56_MB_INTER_V1_GF:

        case VP56_MB_INTER_V2_GF:

            for (b=0; b<b_max; b++) {

                int x_off = b==1 || b==3 ? 8 : 0;

                int y_off = b==2 || b==3 ? 8 : 0;

                plane = ff_vp56_b2p[b+ab];

                vp56_mc(s, b, plane, frame_ref->data[plane], s->stride[plane],

                        16*col+x_off, 16*row+y_off);

                s->dsp.idct_add(frame_current->data[plane] + s->block_offset[b],

                                s->stride[plane], s->block_coeff[b]);

            }

            break;

    }

}
