static int mpeg4_decode_mb(MpegEncContext *s, int16_t block[6][64])

{

    Mpeg4DecContext *ctx = (Mpeg4DecContext *)s;

    int cbpc, cbpy, i, cbp, pred_x, pred_y, mx, my, dquant;

    int16_t *mot_val;

    static const int8_t quant_tab[4] = { -1, -2, 1, 2 };

    const int xy = s->mb_x + s->mb_y * s->mb_stride;



    av_assert2(s->h263_pred);



    if (s->pict_type == AV_PICTURE_TYPE_P ||

        s->pict_type == AV_PICTURE_TYPE_S) {

        do {

            if (get_bits1(&s->gb)) {

                /* skip mb */

                s->mb_intra = 0;

                for (i = 0; i < 6; i++)

                    s->block_last_index[i] = -1;

                s->mv_dir  = MV_DIR_FORWARD;

                s->mv_type = MV_TYPE_16X16;

                if (s->pict_type == AV_PICTURE_TYPE_S &&

                    ctx->vol_sprite_usage == GMC_SPRITE) {

                    s->current_picture.mb_type[xy] = MB_TYPE_SKIP  |

                                                     MB_TYPE_GMC   |

                                                     MB_TYPE_16x16 |

                                                     MB_TYPE_L0;

                    s->mcsel       = 1;

                    s->mv[0][0][0] = get_amv(ctx, 0);

                    s->mv[0][0][1] = get_amv(ctx, 1);

                    s->mb_skipped  = 0;

                } else {

                    s->current_picture.mb_type[xy] = MB_TYPE_SKIP  |

                                                     MB_TYPE_16x16 |

                                                     MB_TYPE_L0;

                    s->mcsel       = 0;

                    s->mv[0][0][0] = 0;

                    s->mv[0][0][1] = 0;

                    s->mb_skipped  = 1;


                goto end;


            cbpc = get_vlc2(&s->gb, ff_h263_inter_MCBPC_vlc.table, INTER_MCBPC_VLC_BITS, 2);

            if (cbpc < 0) {


                       "mcbpc damaged at %d %d\n", s->mb_x, s->mb_y);

                return -1;


        } while (cbpc == 20);



        s->bdsp.clear_blocks(s->block[0]);

        dquant      = cbpc & 8;

        s->mb_intra = ((cbpc & 4) != 0);

        if (s->mb_intra)

            goto intra;



        if (s->pict_type == AV_PICTURE_TYPE_S &&

            ctx->vol_sprite_usage == GMC_SPRITE && (cbpc & 16) == 0)

            s->mcsel = get_bits1(&s->gb);

        else

            s->mcsel = 0;

        cbpy = get_vlc2(&s->gb, ff_h263_cbpy_vlc.table, CBPY_VLC_BITS, 1) ^ 0x0F;








        cbp = (cbpc & 3) | (cbpy << 2);

        if (dquant)

            ff_set_qscale(s, s->qscale + quant_tab[get_bits(&s->gb, 2)]);

        if ((!s->progressive_sequence) &&

            (cbp || (s->workaround_bugs & FF_BUG_XVID_ILACE)))

            s->interlaced_dct = get_bits1(&s->gb);



        s->mv_dir = MV_DIR_FORWARD;

        if ((cbpc & 16) == 0) {

            if (s->mcsel) {

                s->current_picture.mb_type[xy] = MB_TYPE_GMC   |

                                                 MB_TYPE_16x16 |

                                                 MB_TYPE_L0;

                /* 16x16 global motion prediction */

                s->mv_type     = MV_TYPE_16X16;

                mx             = get_amv(ctx, 0);

                my             = get_amv(ctx, 1);

                s->mv[0][0][0] = mx;

                s->mv[0][0][1] = my;

            } else if ((!s->progressive_sequence) && get_bits1(&s->gb)) {

                s->current_picture.mb_type[xy] = MB_TYPE_16x8 |

                                                 MB_TYPE_L0   |

                                                 MB_TYPE_INTERLACED;

                /* 16x8 field motion prediction */

                s->mv_type = MV_TYPE_FIELD;



                s->field_select[0][0] = get_bits1(&s->gb);

                s->field_select[0][1] = get_bits1(&s->gb);



                ff_h263_pred_motion(s, 0, 0, &pred_x, &pred_y);



                for (i = 0; i < 2; i++) {

                    mx = ff_h263_decode_motion(s, pred_x, s->f_code);

                    if (mx >= 0xffff)

                        return -1;



                    my = ff_h263_decode_motion(s, pred_y / 2, s->f_code);

                    if (my >= 0xffff)

                        return -1;



                    s->mv[0][i][0] = mx;

                    s->mv[0][i][1] = my;


            } else {

                s->current_picture.mb_type[xy] = MB_TYPE_16x16 | MB_TYPE_L0;

                /* 16x16 motion prediction */

                s->mv_type = MV_TYPE_16X16;

                ff_h263_pred_motion(s, 0, 0, &pred_x, &pred_y);

                mx = ff_h263_decode_motion(s, pred_x, s->f_code);



                if (mx >= 0xffff)

                    return -1;



                my = ff_h263_decode_motion(s, pred_y, s->f_code);



                if (my >= 0xffff)

                    return -1;

                s->mv[0][0][0] = mx;

                s->mv[0][0][1] = my;


        } else {

            s->current_picture.mb_type[xy] = MB_TYPE_8x8 | MB_TYPE_L0;

            s->mv_type                     = MV_TYPE_8X8;

            for (i = 0; i < 4; i++) {

                mot_val = ff_h263_pred_motion(s, i, 0, &pred_x, &pred_y);

                mx      = ff_h263_decode_motion(s, pred_x, s->f_code);

                if (mx >= 0xffff)

                    return -1;



                my = ff_h263_decode_motion(s, pred_y, s->f_code);

                if (my >= 0xffff)

                    return -1;

                s->mv[0][i][0] = mx;

                s->mv[0][i][1] = my;

                mot_val[0]     = mx;

                mot_val[1]     = my;



    } else if (s->pict_type == AV_PICTURE_TYPE_B) {

        int modb1;   // first bit of modb

        int modb2;   // second bit of modb

        int mb_type;



        s->mb_intra = 0;  // B-frames never contain intra blocks

        s->mcsel    = 0;  //      ...               true gmc blocks



        if (s->mb_x == 0) {

            for (i = 0; i < 2; i++) {

                s->last_mv[i][0][0] =

                s->last_mv[i][0][1] =

                s->last_mv[i][1][0] =

                s->last_mv[i][1][1] = 0;




            ff_thread_await_progress(&s->next_picture_ptr->tf, s->mb_y, 0);




        /* if we skipped it in the future P Frame than skip it now too */

        s->mb_skipped = s->next_picture.mbskip_table[s->mb_y * s->mb_stride + s->mb_x];  // Note, skiptab=0 if last was GMC



        if (s->mb_skipped) {

            /* skip mb */

            for (i = 0; i < 6; i++)

                s->block_last_index[i] = -1;



            s->mv_dir      = MV_DIR_FORWARD;

            s->mv_type     = MV_TYPE_16X16;

            s->mv[0][0][0] =

            s->mv[0][0][1] =

            s->mv[1][0][0] =

            s->mv[1][0][1] = 0;

            s->current_picture.mb_type[xy] = MB_TYPE_SKIP  |

                                             MB_TYPE_16x16 |

                                             MB_TYPE_L0;

            goto end;




        modb1 = get_bits1(&s->gb);

        if (modb1) {

            // like MB_TYPE_B_DIRECT but no vectors coded

            mb_type = MB_TYPE_DIRECT2 | MB_TYPE_SKIP | MB_TYPE_L0L1;

            cbp     = 0;

        } else {

            modb2   = get_bits1(&s->gb);

            mb_type = get_vlc2(&s->gb, mb_type_b_vlc.table, MB_TYPE_B_VLC_BITS, 1);

            if (mb_type < 0) {

                av_log(s->avctx, AV_LOG_ERROR, "illegal MB_type\n");

                return -1;


            mb_type = mb_type_b_map[mb_type];

            if (modb2) {

                cbp = 0;

            } else {

                s->bdsp.clear_blocks(s->block[0]);

                cbp = get_bits(&s->gb, 6);




            if ((!IS_DIRECT(mb_type)) && cbp) {

                if (get_bits1(&s->gb))

                    ff_set_qscale(s, s->qscale + get_bits1(&s->gb) * 4 - 2);




            if (!s->progressive_sequence) {

                if (cbp)

                    s->interlaced_dct = get_bits1(&s->gb);



                if (!IS_DIRECT(mb_type) && get_bits1(&s->gb)) {

                    mb_type |= MB_TYPE_16x8 | MB_TYPE_INTERLACED;

                    mb_type &= ~MB_TYPE_16x16;



                    if (USES_LIST(mb_type, 0)) {

                        s->field_select[0][0] = get_bits1(&s->gb);

                        s->field_select[0][1] = get_bits1(&s->gb);


                    if (USES_LIST(mb_type, 1)) {

                        s->field_select[1][0] = get_bits1(&s->gb);

                        s->field_select[1][1] = get_bits1(&s->gb);






            s->mv_dir = 0;

            if ((mb_type & (MB_TYPE_DIRECT2 | MB_TYPE_INTERLACED)) == 0) {

                s->mv_type = MV_TYPE_16X16;



                if (USES_LIST(mb_type, 0)) {

                    s->mv_dir = MV_DIR_FORWARD;



                    mx = ff_h263_decode_motion(s, s->last_mv[0][0][0], s->f_code);

                    my = ff_h263_decode_motion(s, s->last_mv[0][0][1], s->f_code);

                    s->last_mv[0][1][0] =

                    s->last_mv[0][0][0] =

                    s->mv[0][0][0]      = mx;

                    s->last_mv[0][1][1] =

                    s->last_mv[0][0][1] =

                    s->mv[0][0][1]      = my;




                if (USES_LIST(mb_type, 1)) {

                    s->mv_dir |= MV_DIR_BACKWARD;



                    mx = ff_h263_decode_motion(s, s->last_mv[1][0][0], s->b_code);

                    my = ff_h263_decode_motion(s, s->last_mv[1][0][1], s->b_code);

                    s->last_mv[1][1][0] =

                    s->last_mv[1][0][0] =

                    s->mv[1][0][0]      = mx;

                    s->last_mv[1][1][1] =

                    s->last_mv[1][0][1] =

                    s->mv[1][0][1]      = my;


            } else if (!IS_DIRECT(mb_type)) {

                s->mv_type = MV_TYPE_FIELD;



                if (USES_LIST(mb_type, 0)) {

                    s->mv_dir = MV_DIR_FORWARD;



                    for (i = 0; i < 2; i++) {

                        mx = ff_h263_decode_motion(s, s->last_mv[0][i][0], s->f_code);

                        my = ff_h263_decode_motion(s, s->last_mv[0][i][1] / 2, s->f_code);

                        s->last_mv[0][i][0] =

                        s->mv[0][i][0]      = mx;

                        s->last_mv[0][i][1] = (s->mv[0][i][1] = my) * 2;





                if (USES_LIST(mb_type, 1)) {

                    s->mv_dir |= MV_DIR_BACKWARD;



                    for (i = 0; i < 2; i++) {

                        mx = ff_h263_decode_motion(s, s->last_mv[1][i][0], s->b_code);

                        my = ff_h263_decode_motion(s, s->last_mv[1][i][1] / 2, s->b_code);

                        s->last_mv[1][i][0] =

                        s->mv[1][i][0]      = mx;

                        s->last_mv[1][i][1] = (s->mv[1][i][1] = my) * 2;







        if (IS_DIRECT(mb_type)) {

            if (IS_SKIP(mb_type)) {

                mx =

                my = 0;

            } else {

                mx = ff_h263_decode_motion(s, 0, 1);

                my = ff_h263_decode_motion(s, 0, 1);




            s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD | MV_DIRECT;

            mb_type  |= ff_mpeg4_set_direct_mv(s, mx, my);


        s->current_picture.mb_type[xy] = mb_type;

    } else { /* I-Frame */

        do {

            cbpc = get_vlc2(&s->gb, ff_h263_intra_MCBPC_vlc.table, INTRA_MCBPC_VLC_BITS, 2);

            if (cbpc < 0) {


                       "I cbpc damaged at %d %d\n", s->mb_x, s->mb_y);

                return -1;


        } while (cbpc == 8);



        dquant = cbpc & 4;

        s->mb_intra = 1;



intra:

        s->ac_pred = get_bits1(&s->gb);

        if (s->ac_pred)

            s->current_picture.mb_type[xy] = MB_TYPE_INTRA | MB_TYPE_ACPRED;

        else

            s->current_picture.mb_type[xy] = MB_TYPE_INTRA;



        cbpy = get_vlc2(&s->gb, ff_h263_cbpy_vlc.table, CBPY_VLC_BITS, 1);



                   "I cbpy damaged at %d %d\n", s->mb_x, s->mb_y);

            return -1;


        cbp = (cbpc & 3) | (cbpy << 2);



        ctx->use_intra_dc_vlc = s->qscale < ctx->intra_dc_threshold;



        if (dquant)

            ff_set_qscale(s, s->qscale + quant_tab[get_bits(&s->gb, 2)]);



        if (!s->progressive_sequence)

            s->interlaced_dct = get_bits1(&s->gb);



        s->bdsp.clear_blocks(s->block[0]);

        /* decode each block */

        for (i = 0; i < 6; i++) {

            if (mpeg4_decode_block(ctx, block[i], i, cbp & 32, 1, 0) < 0)

                return -1;

            cbp += cbp;


        goto end;




    /* decode each block */

    for (i = 0; i < 6; i++) {

        if (mpeg4_decode_block(ctx, block[i], i, cbp & 32, 0, 0) < 0)

            return -1;

        cbp += cbp;




end:

    /* per-MB end of slice check */

    if (s->codec_id == AV_CODEC_ID_MPEG4) {

        int next = mpeg4_is_resync(ctx);

        if (next) {

            if        (s->mb_x + s->mb_y*s->mb_width + 1 >  next && (s->avctx->err_recognition & AV_EF_AGGRESSIVE)) {

                return -1;

            } else if (s->mb_x + s->mb_y*s->mb_width + 1 >= next)

                return SLICE_END;



            if (s->pict_type == AV_PICTURE_TYPE_B) {

                const int delta= s->mb_x + 1 == s->mb_width ? 2 : 1;

                ff_thread_await_progress(&s->next_picture_ptr->tf,

                                         (s->mb_x + delta >= s->mb_width)

                                         ? FFMIN(s->mb_y + 1, s->mb_height - 1)

                                         : s->mb_y, 0);

                if (s->next_picture.mbskip_table[xy + delta])

                    return SLICE_OK;




            return SLICE_END;





    return SLICE_OK;
