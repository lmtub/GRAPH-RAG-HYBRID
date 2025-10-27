void vp78_decode_mv_mb_modes(AVCodecContext *avctx, VP8Frame *curframe,

                                    VP8Frame *prev_frame, int is_vp7)

{

    VP8Context *s = avctx->priv_data;

    int mb_x, mb_y;



    s->mv_min.y = -MARGIN;

    s->mv_max.y = ((s->mb_height - 1) << 6) + MARGIN;

    for (mb_y = 0; mb_y < s->mb_height; mb_y++) {

        VP8Macroblock *mb = s->macroblocks_base +

                            ((s->mb_width + 1) * (mb_y + 1) + 1);

        int mb_xy = mb_y * s->mb_width;



        AV_WN32A(s->intra4x4_pred_mode_left, DC_PRED * 0x01010101);



        s->mv_min.x = -MARGIN;

        s->mv_max.x = ((s->mb_width - 1) << 6) + MARGIN;

        for (mb_x = 0; mb_x < s->mb_width; mb_x++, mb_xy++, mb++) {

            if (mb_y == 0)

                AV_WN32A((mb - s->mb_width - 1)->intra4x4_pred_mode_top,

                         DC_PRED * 0x01010101);

            decode_mb_mode(s, mb, mb_x, mb_y, curframe->seg_map->data + mb_xy,

                           prev_frame && prev_frame->seg_map ?

                           prev_frame->seg_map->data + mb_xy : NULL, 1, is_vp7);

            s->mv_min.x -= 64;

            s->mv_max.x -= 64;

        }

        s->mv_min.y -= 64;

        s->mv_max.y -= 64;

    }

}
