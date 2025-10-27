static int svq1_decode_delta_block(MpegEncContext *s, GetBitContext *bitbuf,

                                   uint8_t *current, uint8_t *previous,

                                   int pitch, svq1_pmv *motion, int x, int y)

{

    uint32_t block_type;

    int result = 0;



    /* get block type */

    block_type = get_vlc2(bitbuf, svq1_block_type.table, 2, 2);



    /* reset motion vectors */

    if (block_type == SVQ1_BLOCK_SKIP || block_type == SVQ1_BLOCK_INTRA) {

        motion[0].x         =

        motion[0].y         =

        motion[x / 8 + 2].x =

        motion[x / 8 + 2].y =

        motion[x / 8 + 3].x =

        motion[x / 8 + 3].y = 0;

    }



    switch (block_type) {

    case SVQ1_BLOCK_SKIP:

        svq1_skip_block(current, previous, pitch, x, y);

        break;



    case SVQ1_BLOCK_INTER:

        result = svq1_motion_inter_block(s, bitbuf, current, previous,

                                         pitch, motion, x, y);



        if (result != 0) {

            av_dlog(s->avctx, "Error in svq1_motion_inter_block %i\n", result);

            break;

        }

        result = svq1_decode_block_non_intra(bitbuf, current, pitch);

        break;



    case SVQ1_BLOCK_INTER_4V:

        result = svq1_motion_inter_4v_block(s, bitbuf, current, previous,

                                            pitch, motion, x, y);



        if (result != 0) {

            av_dlog(s->avctx,

                    "Error in svq1_motion_inter_4v_block %i\n", result);

            break;

        }

        result = svq1_decode_block_non_intra(bitbuf, current, pitch);

        break;



    case SVQ1_BLOCK_INTRA:

        result = svq1_decode_block_intra(bitbuf, current, pitch);

        break;

    }



    return result;

}
