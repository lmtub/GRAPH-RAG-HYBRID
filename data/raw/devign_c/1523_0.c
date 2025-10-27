static int bethsoftvid_decode_frame(AVCodecContext *avctx,

                              void *data, int *data_size,

                              AVPacket *avpkt)

{

    BethsoftvidContext * vid = avctx->priv_data;

    char block_type;

    uint8_t * dst;

    uint8_t * frame_end;

    int remaining = avctx->width;          // number of bytes remaining on a line

    const int wrap_to_next_line = vid->frame.linesize[0] - avctx->width;

    int code;

    int yoffset;



    if (avctx->reget_buffer(avctx, &vid->frame)) {

        av_log(avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

        return -1;

    }



    bytestream2_init(&vid->g, avpkt->data, avpkt->size);

    dst = vid->frame.data[0];

    frame_end = vid->frame.data[0] + vid->frame.linesize[0] * avctx->height;



    switch(block_type = bytestream2_get_byte(&vid->g)){

        case PALETTE_BLOCK: {

            int ret;

            *data_size = 0;

            if ((ret = set_palette(vid)) < 0) {

                av_log(avctx, AV_LOG_ERROR, "error reading palette\n");

                return ret;

            }

            return bytestream2_tell(&vid->g);

        }

        case VIDEO_YOFF_P_FRAME:

            yoffset = bytestream2_get_le16(&vid->g);

            if(yoffset >= avctx->height)

                return -1;

            dst += vid->frame.linesize[0] * yoffset;

    }



    // main code

    while((code = bytestream2_get_byte(&vid->g))){

        int length = code & 0x7f;



        // copy any bytes starting at the current position, and ending at the frame width

        while(length > remaining){

            if(code < 0x80)

                bytestream2_get_buffer(&vid->g, dst, remaining);

            else if(block_type == VIDEO_I_FRAME)

                memset(dst, bytestream2_peek_byte(&vid->g), remaining);

            length -= remaining;      // decrement the number of bytes to be copied

            dst += remaining + wrap_to_next_line;    // skip over extra bytes at end of frame

            remaining = avctx->width;

            if(dst == frame_end)

                goto end;

        }



        // copy any remaining bytes after / if line overflows

        if(code < 0x80)

            bytestream2_get_buffer(&vid->g, dst, length);

        else if(block_type == VIDEO_I_FRAME)

            memset(dst, bytestream2_get_byte(&vid->g), length);

        remaining -= length;

        dst += length;

    }

    end:



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = vid->frame;



    return avpkt->size;

}
