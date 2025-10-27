static int decode_frame(AVCodecContext *avctx, void *data, int *got_frame, AVPacket *pkt)

{

    BinkContext * const c = avctx->priv_data;

    GetBitContext gb;

    int plane, plane_idx, ret;

    int bits_count = pkt->size << 3;



    if (c->version > 'b') {

        if(c->pic->data[0])

            avctx->release_buffer(avctx, c->pic);



        if ((ret = ff_get_buffer(avctx, c->pic)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

            return ret;

        }

    } else {

        if ((ret = avctx->reget_buffer(avctx, c->pic)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

            return ret;

        }

    }



    init_get_bits(&gb, pkt->data, bits_count);

    if (c->has_alpha) {

        if (c->version >= 'i')

            skip_bits_long(&gb, 32);

        if ((ret = bink_decode_plane(c, &gb, 3, 0)) < 0)

            return ret;

    }

    if (c->version >= 'i')

        skip_bits_long(&gb, 32);



    for (plane = 0; plane < 3; plane++) {

        plane_idx = (!plane || !c->swap_planes) ? plane : (plane ^ 3);



        if (c->version > 'b') {

            if ((ret = bink_decode_plane(c, &gb, plane_idx, !!plane)) < 0)

                return ret;

        } else {

            if ((ret = binkb_decode_plane(c, &gb, plane_idx, !pkt->pts, !!plane)) < 0)

                return ret;

        }

        if (get_bits_count(&gb) >= bits_count)

            break;

    }

    emms_c();



    *got_frame = 1;

    *(AVFrame*)data = *c->pic;



    if (c->version > 'b')

        FFSWAP(AVFrame*, c->pic, c->last);



    /* always report that the buffer was completely consumed */

    return pkt->size;

}
