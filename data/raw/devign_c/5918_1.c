static int decode_frame(AVCodecContext *avctx, void *data, int *data_size, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    VBDecContext * const c = avctx->priv_data;

    uint8_t *outptr, *srcptr;

    int i, j;

    int flags;

    uint32_t size;

    int rest = buf_size;

    int offset = 0;



    if(c->pic.data[0])

        avctx->release_buffer(avctx, &c->pic);

    c->pic.reference = 3;

    if(avctx->get_buffer(avctx, &c->pic) < 0){

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    c->stream = buf;

    flags = bytestream_get_le16(&c->stream);

    rest -= 2;



    if(flags & VB_HAS_GMC){

        i = (int16_t)bytestream_get_le16(&c->stream);

        j = (int16_t)bytestream_get_le16(&c->stream);

        offset = i + j * avctx->width;

        rest -= 4;

    }

    if(flags & VB_HAS_VIDEO){

        size = bytestream_get_le32(&c->stream);

        if(size > rest){

            av_log(avctx, AV_LOG_ERROR, "Frame size is too big\n");

            return -1;

        }

        vb_decode_framedata(c, c->stream, size, offset);

        c->stream += size - 4;

        rest -= size;

    }

    if(flags & VB_HAS_PALETTE){

        size = bytestream_get_le32(&c->stream);

        if(size > rest){

            av_log(avctx, AV_LOG_ERROR, "Palette size is too big\n");

            return -1;

        }

        vb_decode_palette(c, size);

        rest -= size;

    }



    memcpy(c->pic.data[1], c->pal, AVPALETTE_SIZE);

    c->pic.palette_has_changed = flags & VB_HAS_PALETTE;



    outptr = c->pic.data[0];

    srcptr = c->frame;



    for(i = 0; i < avctx->height; i++){

        memcpy(outptr, srcptr, avctx->width);

        srcptr += avctx->width;

        outptr += c->pic.linesize[0];

    }



    FFSWAP(uint8_t*, c->frame, c->prev_frame);



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = c->pic;



    /* always report that the buffer was completely consumed */

    return buf_size;

}
