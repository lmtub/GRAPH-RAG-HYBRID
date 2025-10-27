static av_always_inline void vc1_apply_p_h_loop_filter(VC1Context *v, int block_num)

{

    MpegEncContext *s  = &v->s;

    int mb_cbp         = v->cbp[s->mb_x - 1 - s->mb_stride],

        block_cbp      = mb_cbp      >> (block_num * 4), right_cbp,

        mb_is_intra    = v->is_intra[s->mb_x - 1 - s->mb_stride],

        block_is_intra = mb_is_intra >> (block_num * 4), right_is_intra;

    int idx, linesize  = block_num > 3 ? s->uvlinesize : s->linesize, ttblk;

    uint8_t *dst;



    if (block_num > 3) {

        dst = s->dest[block_num - 3] - 8 * linesize;

    } else {

        dst = s->dest[0] + (block_num & 1) * 8 + ((block_num & 2) * 4 - 16) * linesize - 8;

    }



    if (s->mb_x != s->mb_width || !(block_num & 5)) {

        int16_t (*mv)[2];



        if (block_num > 3) {

            right_cbp      = v->cbp[s->mb_x - s->mb_stride] >> (block_num * 4);

            right_is_intra = v->is_intra[s->mb_x - s->mb_stride] >> (block_num * 4);

            mv             = &v->luma_mv[s->mb_x - s->mb_stride - 1];

        } else {

            right_cbp      = (block_num & 1) ? (v->cbp[s->mb_x - s->mb_stride]      >> ((block_num - 1) * 4))

                                             : (mb_cbp                              >> ((block_num + 1) * 4));

            right_is_intra = (block_num & 1) ? (v->is_intra[s->mb_x - s->mb_stride] >> ((block_num - 1) * 4))

                                             : (mb_is_intra                         >> ((block_num + 1) * 4));

            mv             = &s->current_picture.motion_val[0][s->block_index[block_num] - s->b8_stride * 2 - 2];

        }

        if (block_is_intra & 1 || right_is_intra & 1 || mv[0][0] != mv[1][0] || mv[0][1] != mv[1][1]) {

            v->vc1dsp.vc1_h_loop_filter8(dst, linesize, v->pq);

        } else {

            idx = ((right_cbp >> 1) | block_cbp) & 5; // FIXME check

            if (idx == 5) {

                v->vc1dsp.vc1_h_loop_filter8(dst, linesize, v->pq);

            } else if (idx) {

                if (idx == 1)

                    v->vc1dsp.vc1_h_loop_filter4(dst + 4 * linesize, linesize, v->pq);

                else

                    v->vc1dsp.vc1_h_loop_filter4(dst,                linesize, v->pq);

            }

        }

    }



    dst -= 4;

    ttblk = (v->ttblk[s->mb_x - s->mb_stride - 1] >> (block_num * 4)) & 0xf;

    if (ttblk == TT_4X4 || ttblk == TT_4X8) {

        idx = (block_cbp | (block_cbp >> 1)) & 5;

        if (idx == 5) {

            v->vc1dsp.vc1_h_loop_filter8(dst, linesize, v->pq);

        } else if (idx) {

            if (idx == 1)

                v->vc1dsp.vc1_h_loop_filter4(dst + linesize * 4, linesize, v->pq);

            else

                v->vc1dsp.vc1_h_loop_filter4(dst,                linesize, v->pq);

        }

    }

}
