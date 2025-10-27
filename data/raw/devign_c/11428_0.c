void ff_vc1_decode_blocks(VC1Context *v)

{



    v->s.esc3_level_length = 0;

    if (v->x8_type) {

        ff_intrax8_decode_picture(&v->x8, 2*v->pq + v->halfpq, v->pq * !v->pquantizer);



        ff_er_add_slice(&v->s.er, 0, 0,

                        (v->s.mb_x >> 1) - 1, (v->s.mb_y >> 1) - 1,

                        ER_MB_END);

    } else {

        v->cur_blk_idx     =  0;

        v->left_blk_idx    = -1;

        v->topleft_blk_idx =  1;

        v->top_blk_idx     =  2;

        switch (v->s.pict_type) {

        case AV_PICTURE_TYPE_I:

            if (v->profile == PROFILE_ADVANCED)

                vc1_decode_i_blocks_adv(v);

            else

                vc1_decode_i_blocks(v);

            break;

        case AV_PICTURE_TYPE_P:

            if (v->p_frame_skipped)

                vc1_decode_skip_blocks(v);

            else

                vc1_decode_p_blocks(v);

            break;

        case AV_PICTURE_TYPE_B:

            if (v->bi_type) {

                if (v->profile == PROFILE_ADVANCED)

                    vc1_decode_i_blocks_adv(v);

                else

                    vc1_decode_i_blocks(v);

            } else

                vc1_decode_b_blocks(v);

            break;

        }

    }

}
