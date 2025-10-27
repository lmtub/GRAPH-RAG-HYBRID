static int encode_picture_ls(AVCodecContext *avctx, AVPacket *pkt,

                             const AVFrame *pict, int *got_packet)

{

    const AVFrame *const p = pict;

    const int near         = avctx->prediction_method;

    PutBitContext pb, pb2;

    GetBitContext gb;

    uint8_t *buf2, *zero, *cur, *last;

    JLSState *state;

    int i, size, ret;

    int comps;



    if (avctx->pix_fmt == AV_PIX_FMT_GRAY8 ||

        avctx->pix_fmt == AV_PIX_FMT_GRAY16)

        comps = 1;

    else

        comps = 3;



    if ((ret = ff_alloc_packet(pkt, avctx->width * avctx->height * comps * 4 +

                               FF_MIN_BUFFER_SIZE)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet.\n");

        return ret;

    }



    buf2 = av_malloc(pkt->size);



    init_put_bits(&pb, pkt->data, pkt->size);

    init_put_bits(&pb2, buf2, pkt->size);



    /* write our own JPEG header, can't use mjpeg_picture_header */

    put_marker(&pb, SOI);

    put_marker(&pb, SOF48);

    put_bits(&pb, 16, 8 + comps * 3); // header size depends on components

    put_bits(&pb, 8, (avctx->pix_fmt == AV_PIX_FMT_GRAY16) ? 16 : 8);  // bpp

    put_bits(&pb, 16, avctx->height);

    put_bits(&pb, 16, avctx->width);

    put_bits(&pb, 8, comps);          // components

    for (i = 1; i <= comps; i++) {

        put_bits(&pb, 8, i);     // component ID

        put_bits(&pb, 8, 0x11);  // subsampling: none

        put_bits(&pb, 8, 0);     // Tiq, used by JPEG-LS ext

    }



    put_marker(&pb, SOS);

    put_bits(&pb, 16, 6 + comps * 2);

    put_bits(&pb, 8, comps);

    for (i = 1; i <= comps; i++) {

        put_bits(&pb, 8, i);   // component ID

        put_bits(&pb, 8, 0);   // mapping index: none

    }

    put_bits(&pb, 8, near);

    put_bits(&pb, 8, (comps > 1) ? 1 : 0);  // interleaving: 0 - plane, 1 - line

    put_bits(&pb, 8, 0);  // point transform: none



    state = av_mallocz(sizeof(JLSState));

    /* initialize JPEG-LS state from JPEG parameters */

    state->near = near;

    state->bpp  = (avctx->pix_fmt == AV_PIX_FMT_GRAY16) ? 16 : 8;

    ff_jpegls_reset_coding_parameters(state, 0);

    ff_jpegls_init_state(state);



    ls_store_lse(state, &pb);



    zero = av_mallocz(p->linesize[0]);

    last = zero;

    cur  = p->data[0];

    if (avctx->pix_fmt == AV_PIX_FMT_GRAY8) {

        int t = 0;



        for (i = 0; i < avctx->height; i++) {

            ls_encode_line(state, &pb2, last, cur, t, avctx->width, 1, 0, 8);

            t    = last[0];

            last = cur;

            cur += p->linesize[0];

        }

    } else if (avctx->pix_fmt == AV_PIX_FMT_GRAY16) {

        int t = 0;



        for (i = 0; i < avctx->height; i++) {

            ls_encode_line(state, &pb2, last, cur, t, avctx->width, 1, 0, 16);

            t    = *((uint16_t *)last);

            last = cur;

            cur += p->linesize[0];

        }

    } else if (avctx->pix_fmt == AV_PIX_FMT_RGB24) {

        int j, width;

        int Rc[3] = { 0, 0, 0 };



        width = avctx->width * 3;

        for (i = 0; i < avctx->height; i++) {

            for (j = 0; j < 3; j++) {

                ls_encode_line(state, &pb2, last + j, cur + j, Rc[j],

                               width, 3, j, 8);

                Rc[j] = last[j];

            }

            last = cur;

            cur += p->linesize[0];

        }

    } else if (avctx->pix_fmt == AV_PIX_FMT_BGR24) {

        int j, width;

        int Rc[3] = { 0, 0, 0 };



        width = avctx->width * 3;

        for (i = 0; i < avctx->height; i++) {

            for (j = 2; j >= 0; j--) {

                ls_encode_line(state, &pb2, last + j, cur + j, Rc[j],

                               width, 3, j, 8);

                Rc[j] = last[j];

            }

            last = cur;

            cur += p->linesize[0];

        }

    }



    av_free(zero);

    av_free(state);



    /* the specification says that after doing 0xff escaping unused bits in

     * the last byte must be set to 0, so just append 7 "optional" zero-bits

     * to avoid special-casing. */

    put_bits(&pb2, 7, 0);

    size = put_bits_count(&pb2);

    flush_put_bits(&pb2);

    /* do escape coding */

    init_get_bits(&gb, buf2, size);

    size -= 7;

    while (get_bits_count(&gb) < size) {

        int v;

        v = get_bits(&gb, 8);

        put_bits(&pb, 8, v);

        if (v == 0xFF) {

            v = get_bits(&gb, 7);

            put_bits(&pb, 8, v);

        }

    }

    avpriv_align_put_bits(&pb);

    av_free(buf2);



    /* End of image */

    put_marker(&pb, EOI);

    flush_put_bits(&pb);



    emms_c();



    pkt->size   = put_bits_count(&pb) >> 3;

    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;

    return 0;

}
