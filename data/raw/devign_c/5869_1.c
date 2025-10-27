static int vp3_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    Vp3DecodeContext *s = avctx->priv_data;

    GetBitContext gb;

    static int counter = 0;

    int i;



    init_get_bits(&gb, buf, buf_size * 8);



    if (s->theora && get_bits1(&gb))

    {

        av_log(avctx, AV_LOG_ERROR, "Header packet passed to frame decoder, skipping\n");

        return -1;

    }



    s->keyframe = !get_bits1(&gb);

    if (!s->theora)

        skip_bits(&gb, 1);

    for (i = 0; i < 3; i++)

        s->last_qps[i] = s->qps[i];



    s->nqps=0;

    do{

        s->qps[s->nqps++]= get_bits(&gb, 6);

    } while(s->theora >= 0x030200 && s->nqps<3 && get_bits1(&gb));

    for (i = s->nqps; i < 3; i++)

        s->qps[i] = -1;



    if (s->avctx->debug & FF_DEBUG_PICT_INFO)

        av_log(s->avctx, AV_LOG_INFO, " VP3 %sframe #%d: Q index = %d\n",

            s->keyframe?"key":"", counter, s->qps[0]);

    counter++;



    if (s->qps[0] != s->last_qps[0])

        init_loop_filter(s);



    for (i = 0; i < s->nqps; i++)

        // reinit all dequantizers if the first one changed, because

        // the DC of the first quantizer must be used for all matrices

        if (s->qps[i] != s->last_qps[i] || s->qps[0] != s->last_qps[0])

            init_dequantizer(s, i);



    if (avctx->skip_frame >= AVDISCARD_NONKEY && !s->keyframe)

        return buf_size;



    s->current_frame.reference = 3;

    if (avctx->get_buffer(avctx, &s->current_frame) < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    if (s->keyframe) {

        if (!s->theora)

        {

            skip_bits(&gb, 4); /* width code */

            skip_bits(&gb, 4); /* height code */

            if (s->version)

            {

                s->version = get_bits(&gb, 5);

                if (counter == 1)

                    av_log(s->avctx, AV_LOG_DEBUG, "VP version: %d\n", s->version);

            }

        }

        if (s->version || s->theora)

        {

                if (get_bits1(&gb))

                    av_log(s->avctx, AV_LOG_ERROR, "Warning, unsupported keyframe coding type?!\n");

            skip_bits(&gb, 2); /* reserved? */

        }

    } else {

        if (!s->golden_frame.data[0]) {

            av_log(s->avctx, AV_LOG_ERROR, "vp3: first frame not a keyframe\n");

            avctx->release_buffer(avctx, &s->current_frame);

            return -1;

        }

    }



    s->current_frame.qscale_table= s->qscale_table; //FIXME allocate individual tables per AVFrame

    s->current_frame.qstride= 0;



    init_frame(s, &gb);



    if (unpack_superblocks(s, &gb)){

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_superblocks\n");

        return -1;

    }

    if (unpack_modes(s, &gb)){

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_modes\n");

        return -1;

    }

    if (unpack_vectors(s, &gb)){

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_vectors\n");

        return -1;

    }

    if (unpack_block_qpis(s, &gb)){

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_block_qpis\n");

        return -1;

    }

    if (unpack_dct_coeffs(s, &gb)){

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_dct_coeffs\n");

        return -1;

    }



    for (i = 0; i < 3; i++) {

        if (s->flipped_image)

            s->data_offset[i] = 0;

        else

            s->data_offset[i] = ((s->height>>!!i)-1) * s->current_frame.linesize[i];

    }



    s->last_slice_end = 0;

    for (i = 0; i < s->c_superblock_height; i++)

        render_slice(s, i);



    // filter the last row

    for (i = 0; i < 3; i++) {

        int row = (s->height >> (3+!!i)) - 1;

        apply_loop_filter(s, i, row, row+1);

    }

    vp3_draw_horiz_band(s, s->height);



    *data_size=sizeof(AVFrame);

    *(AVFrame*)data= s->current_frame;



    /* release the last frame, if it is allocated and if it is not the

     * golden frame */

    if ((s->last_frame.data[0]) &&

        (s->last_frame.data[0] != s->golden_frame.data[0]))

        avctx->release_buffer(avctx, &s->last_frame);



    /* shuffle frames (last = current) */

    s->last_frame= s->current_frame;



    if (s->keyframe) {

        if (s->golden_frame.data[0])

            avctx->release_buffer(avctx, &s->golden_frame);

        s->golden_frame = s->current_frame;

    }



    s->current_frame.data[0]= NULL; /* ensure that we catch any access to this released frame */



    return buf_size;

}
