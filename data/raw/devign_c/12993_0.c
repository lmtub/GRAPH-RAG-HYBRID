static int check_bits_for_superframe(GetBitContext *orig_gb,

                                     WMAVoiceContext *s)

{

    GetBitContext s_gb, *gb = &s_gb;

    int n, need_bits, bd_idx;

    const struct frame_type_desc *frame_desc;



    /* initialize a copy */

    init_get_bits(gb, orig_gb->buffer, orig_gb->size_in_bits);

    skip_bits_long(gb, get_bits_count(orig_gb));

    av_assert1(get_bits_left(gb) == get_bits_left(orig_gb));



    /* superframe header */

    if (get_bits_left(gb) < 14)

        return 1;

    if (!get_bits1(gb))

        return AVERROR(ENOSYS);           // WMAPro-in-WMAVoice superframe

    if (get_bits1(gb)) skip_bits(gb, 12); // number of  samples in superframe

    if (s->has_residual_lsps) {           // residual LSPs (for all frames)

        if (get_bits_left(gb) < s->sframe_lsp_bitsize)

            return 1;

        skip_bits_long(gb, s->sframe_lsp_bitsize);

    }



    /* frames */

    for (n = 0; n < MAX_FRAMES; n++) {

        int aw_idx_is_ext = 0;



        if (!s->has_residual_lsps) {     // independent LSPs (per-frame)

           if (get_bits_left(gb) < s->frame_lsp_bitsize) return 1;

           skip_bits_long(gb, s->frame_lsp_bitsize);

        }

        bd_idx = s->vbm_tree[get_vlc2(gb, frame_type_vlc.table, 6, 3)];

        if (bd_idx < 0)

            return AVERROR_INVALIDDATA; // invalid frame type VLC code

        frame_desc = &frame_descs[bd_idx];

        if (frame_desc->acb_type == ACB_TYPE_ASYMMETRIC) {

            if (get_bits_left(gb) < s->pitch_nbits)

                return 1;

            skip_bits_long(gb, s->pitch_nbits);

        }

        if (frame_desc->fcb_type == FCB_TYPE_SILENCE) {

            skip_bits(gb, 8);

        } else if (frame_desc->fcb_type == FCB_TYPE_AW_PULSES) {

            int tmp = get_bits(gb, 6);

            if (tmp >= 0x36) {

                skip_bits(gb, 2);

                aw_idx_is_ext = 1;

            }

        }



        /* blocks */

        if (frame_desc->acb_type == ACB_TYPE_HAMMING) {

            need_bits = s->block_pitch_nbits +

                (frame_desc->n_blocks - 1) * s->block_delta_pitch_nbits;

        } else if (frame_desc->fcb_type == FCB_TYPE_AW_PULSES) {

            need_bits = 2 * !aw_idx_is_ext;

        } else

            need_bits = 0;

        need_bits += frame_desc->frame_size;

        if (get_bits_left(gb) < need_bits)

            return 1;

        skip_bits_long(gb, need_bits);

    }



    return 0;

}
