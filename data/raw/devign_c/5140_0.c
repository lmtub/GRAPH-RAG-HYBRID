static int raw_decode(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    int linesize_align = 4;

    RawVideoContext *context = avctx->priv_data;



    AVFrame   *frame   = data;

    AVPicture *picture = data;



    frame->pict_type        = avctx->coded_frame->pict_type;

    frame->interlaced_frame = avctx->coded_frame->interlaced_frame;

    frame->top_field_first = avctx->coded_frame->top_field_first;

    frame->reordered_opaque = avctx->reordered_opaque;

    frame->pkt_pts          = avctx->pkt->pts;

    frame->pkt_pos          = avctx->pkt->pos;



    if(context->tff>=0){

        frame->interlaced_frame = 1;

        frame->top_field_first  = context->tff;

    }



    if(buf_size < context->length - (avctx->pix_fmt==PIX_FMT_PAL8 ? 256*4 : 0))

        return -1;



    if (avctx->width <= 0 || avctx->height <= 0) {

        av_log(avctx, AV_LOG_ERROR, "w/h is invalid\n");

        return AVERROR(EINVAL);

    }



    //2bpp and 4bpp raw in avi and mov (yes this is ugly ...)

    if (context->buffer) {

        int i;

        uint8_t *dst = context->buffer;

        buf_size = context->length - 256*4;

        if (avctx->bits_per_coded_sample == 4){

            for(i=0; 2*i+1 < buf_size && i<avpkt->size; i++){

                dst[2*i+0]= buf[i]>>4;

                dst[2*i+1]= buf[i]&15;

            }

            linesize_align = 8;

        } else {

            for(i=0; 4*i+3 < buf_size && i<avpkt->size; i++){

                dst[4*i+0]= buf[i]>>6;

                dst[4*i+1]= buf[i]>>4&3;

                dst[4*i+2]= buf[i]>>2&3;

                dst[4*i+3]= buf[i]   &3;

            }

            linesize_align = 16;

        }

        buf= dst;

    }



    if(avctx->codec_tag == MKTAG('A', 'V', '1', 'x') ||

       avctx->codec_tag == MKTAG('A', 'V', 'u', 'p'))

        buf += buf_size - context->length;



    avpicture_fill(picture, buf, avctx->pix_fmt, avctx->width, avctx->height);

    if((avctx->pix_fmt==PIX_FMT_PAL8 && buf_size < context->length) ||

       (av_pix_fmt_descriptors[avctx->pix_fmt].flags & PIX_FMT_PSEUDOPAL)) {

        frame->data[1]= context->palette;

    }

    if (avctx->pix_fmt == PIX_FMT_PAL8) {

        const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, NULL);



        if (pal) {

            memcpy(frame->data[1], pal, AVPALETTE_SIZE);

            frame->palette_has_changed = 1;

        }

    }

    if((avctx->pix_fmt==PIX_FMT_BGR24    ||

        avctx->pix_fmt==PIX_FMT_GRAY8    ||

        avctx->pix_fmt==PIX_FMT_RGB555LE ||

        avctx->pix_fmt==PIX_FMT_RGB555BE ||

        avctx->pix_fmt==PIX_FMT_RGB565LE ||

        avctx->pix_fmt==PIX_FMT_MONOWHITE ||

        avctx->pix_fmt==PIX_FMT_PAL8) &&

        FFALIGN(frame->linesize[0], linesize_align)*avctx->height <= buf_size)

        frame->linesize[0] = FFALIGN(frame->linesize[0], linesize_align);



    if(context->flip)

        flip(avctx, picture);



    if (   avctx->codec_tag == MKTAG('Y', 'V', '1', '2')

        || avctx->codec_tag == MKTAG('Y', 'V', '1', '6')

        || avctx->codec_tag == MKTAG('Y', 'V', '2', '4')

        || avctx->codec_tag == MKTAG('Y', 'V', 'U', '9'))

        FFSWAP(uint8_t *, picture->data[1], picture->data[2]);



    if(avctx->codec_tag == AV_RL32("yuv2") &&

       avctx->pix_fmt   == PIX_FMT_YUYV422) {

        int x, y;

        uint8_t *line = picture->data[0];

        for(y = 0; y < avctx->height; y++) {

            for(x = 0; x < avctx->width; x++)

                line[2*x + 1] ^= 0x80;

            line += picture->linesize[0];

        }

    }



    *data_size = sizeof(AVPicture);

    return buf_size;

}
