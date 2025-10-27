static int vmd_decode(VmdVideoContext *s, AVFrame *frame)

{

    int i;

    unsigned int *palette32;

    unsigned char r, g, b;



    GetByteContext gb;



    unsigned char meth;

    unsigned char *dp;   /* pointer to current frame */

    unsigned char *pp;   /* pointer to previous frame */

    unsigned char len;

    int ofs;



    int frame_x, frame_y;

    int frame_width, frame_height;



    frame_x = AV_RL16(&s->buf[6]);

    frame_y = AV_RL16(&s->buf[8]);

    frame_width = AV_RL16(&s->buf[10]) - frame_x + 1;

    frame_height = AV_RL16(&s->buf[12]) - frame_y + 1;

    if (frame_x < 0 || frame_width < 0 ||

        frame_x >= s->avctx->width ||

        frame_width > s->avctx->width ||

        frame_x + frame_width > s->avctx->width) {

        av_log(s->avctx, AV_LOG_ERROR,

               "Invalid horizontal range %d-%d\n",

               frame_x, frame_width);

        return AVERROR_INVALIDDATA;

    }

    if (frame_y < 0 || frame_height < 0 ||

        frame_y >= s->avctx->height ||

        frame_height > s->avctx->height ||

        frame_y + frame_height > s->avctx->height) {

        av_log(s->avctx, AV_LOG_ERROR,

               "Invalid vertical range %d-%d\n",

               frame_x, frame_width);

        return AVERROR_INVALIDDATA;

    }



    if ((frame_width == s->avctx->width && frame_height == s->avctx->height) &&

        (frame_x || frame_y)) {



        s->x_off = frame_x;

        s->y_off = frame_y;

    }

    frame_x -= s->x_off;

    frame_y -= s->y_off;



    /* if only a certain region will be updated, copy the entire previous

     * frame before the decode */

    if (s->prev_frame.data[0] &&

        (frame_x || frame_y || (frame_width != s->avctx->width) ||

        (frame_height != s->avctx->height))) {



        memcpy(frame->data[0], s->prev_frame.data[0],

            s->avctx->height * frame->linesize[0]);

    }



    /* check if there is a new palette */

    bytestream2_init(&gb, s->buf + 16, s->size - 16);

    if (s->buf[15] & 0x02) {

        bytestream2_skip(&gb, 2);

        palette32 = (unsigned int *)s->palette;

        if (bytestream2_get_bytes_left(&gb) >= PALETTE_COUNT * 3) {

            for (i = 0; i < PALETTE_COUNT; i++) {

                r = bytestream2_get_byteu(&gb) * 4;

                g = bytestream2_get_byteu(&gb) * 4;

                b = bytestream2_get_byteu(&gb) * 4;

                palette32[i] = (r << 16) | (g << 8) | (b);

            }

        } else {

            av_log(s->avctx, AV_LOG_ERROR, "Incomplete palette\n");

            return AVERROR_INVALIDDATA;

        }

        s->size -= PALETTE_COUNT * 3 + 2;

    }

    if (s->size > 0) {

        /* originally UnpackFrame in VAG's code */

        bytestream2_init(&gb, gb.buffer, s->buf + s->size - gb.buffer);

        if (bytestream2_get_bytes_left(&gb) < 1)

            return AVERROR_INVALIDDATA;

        meth = bytestream2_get_byteu(&gb);

        if (meth & 0x80) {

            lz_unpack(gb.buffer, bytestream2_get_bytes_left(&gb),

                      s->unpack_buffer, s->unpack_buffer_size);

            meth &= 0x7F;

            bytestream2_init(&gb, s->unpack_buffer, s->unpack_buffer_size);

        }



        dp = &frame->data[0][frame_y * frame->linesize[0] + frame_x];

        pp = &s->prev_frame.data[0][frame_y * s->prev_frame.linesize[0] + frame_x];

        switch (meth) {

        case 1:

            for (i = 0; i < frame_height; i++) {

                ofs = 0;

                do {

                    len = bytestream2_get_byte(&gb);

                    if (len & 0x80) {

                        len = (len & 0x7F) + 1;

                        if (ofs + len > frame_width || bytestream2_get_bytes_left(&gb) < len)

                            return AVERROR_INVALIDDATA;

                        bytestream2_get_buffer(&gb, &dp[ofs], len);

                        ofs += len;

                    } else {

                        /* interframe pixel copy */

                        if (ofs + len + 1 > frame_width || !s->prev_frame.data[0])

                            return AVERROR_INVALIDDATA;

                        memcpy(&dp[ofs], &pp[ofs], len + 1);

                        ofs += len + 1;

                    }

                } while (ofs < frame_width);

                if (ofs > frame_width) {

                    av_log(s->avctx, AV_LOG_ERROR, "VMD video: offset > width (%d > %d)\n",

                        ofs, frame_width);

                    return AVERROR_INVALIDDATA;

                }

                dp += frame->linesize[0];

                pp += s->prev_frame.linesize[0];

            }

            break;



        case 2:

            for (i = 0; i < frame_height; i++) {

                bytestream2_get_buffer(&gb, dp, frame_width);

                dp += frame->linesize[0];

                pp += s->prev_frame.linesize[0];

            }

            break;



        case 3:

            for (i = 0; i < frame_height; i++) {

                ofs = 0;

                do {

                    len = bytestream2_get_byte(&gb);

                    if (len & 0x80) {

                        len = (len & 0x7F) + 1;

                        if (bytestream2_get_byte(&gb) == 0xFF)

                            len = rle_unpack(gb.buffer, &dp[ofs],

                                             len, bytestream2_get_bytes_left(&gb),

                                             frame_width - ofs);

                        else

                            bytestream2_get_buffer(&gb, &dp[ofs], len);

                        bytestream2_skip(&gb, len);

                    } else {

                        /* interframe pixel copy */

                        if (ofs + len + 1 > frame_width || !s->prev_frame.data[0])

                            return AVERROR_INVALIDDATA;

                        memcpy(&dp[ofs], &pp[ofs], len + 1);

                        ofs += len + 1;

                    }

                } while (ofs < frame_width);

                if (ofs > frame_width) {

                    av_log(s->avctx, AV_LOG_ERROR, "VMD video: offset > width (%d > %d)\n",

                        ofs, frame_width);

                    return AVERROR_INVALIDDATA;

                }

                dp += frame->linesize[0];

                pp += s->prev_frame.linesize[0];

            }

            break;

        }

    }

    return 0;

}
