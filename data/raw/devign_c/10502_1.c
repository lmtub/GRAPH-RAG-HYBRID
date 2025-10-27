void decode_mb_mode(VP8Context *s, VP8Macroblock *mb, int mb_x, int mb_y,

                    uint8_t *segment, uint8_t *ref, int layout)

{

    VP56RangeCoder *c = &s->c;



    if (s->segmentation.update_map)

        *segment = vp8_rac_get_tree(c, vp8_segmentid_tree, s->prob->segmentid);

    else if (s->segmentation.enabled)

        *segment = ref ? *ref : *segment;

    mb->segment = *segment;



    mb->skip = s->mbskip_enabled ? vp56_rac_get_prob(c, s->prob->mbskip) : 0;



    if (s->keyframe) {

        mb->mode = vp8_rac_get_tree(c, vp8_pred16x16_tree_intra,

                                    vp8_pred16x16_prob_intra);



        if (mb->mode == MODE_I4x4) {

            decode_intra4x4_modes(s, c, mb, mb_x, 1, layout);

        } else {

            const uint32_t modes = vp8_pred4x4_mode[mb->mode] * 0x01010101u;

            if (s->mb_layout == 1)

                AV_WN32A(mb->intra4x4_pred_mode_top, modes);

            else

                AV_WN32A(s->intra4x4_pred_mode_top + 4 * mb_x, modes);

            AV_WN32A(s->intra4x4_pred_mode_left, modes);

        }



        mb->chroma_pred_mode = vp8_rac_get_tree(c, vp8_pred8x8c_tree,

                                                vp8_pred8x8c_prob_intra);

        mb->ref_frame        = VP56_FRAME_CURRENT;

    } else if (vp56_rac_get_prob_branchy(c, s->prob->intra)) {

        // inter MB, 16.2

        if (vp56_rac_get_prob_branchy(c, s->prob->last))

            mb->ref_frame =

                vp56_rac_get_prob(c, s->prob->golden) ? VP56_FRAME_GOLDEN2 /* altref */

                                                      : VP56_FRAME_GOLDEN;

        else

            mb->ref_frame = VP56_FRAME_PREVIOUS;

        s->ref_count[mb->ref_frame - 1]++;



        // motion vectors, 16.3

        decode_mvs(s, mb, mb_x, mb_y, layout);

    } else {

        // intra MB, 16.1

        mb->mode = vp8_rac_get_tree(c, vp8_pred16x16_tree_inter, s->prob->pred16x16);



        if (mb->mode == MODE_I4x4)

            decode_intra4x4_modes(s, c, mb, mb_x, 0, layout);



        mb->chroma_pred_mode = vp8_rac_get_tree(c, vp8_pred8x8c_tree,

                                                s->prob->pred8x8c);

        mb->ref_frame        = VP56_FRAME_CURRENT;

        mb->partitioning     = VP8_SPLITMVMODE_NONE;

        AV_ZERO32(&mb->bmv[0]);

    }

}
