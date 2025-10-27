static int dds_decode(AVCodecContext *avctx, void *data,

                      int *got_frame, AVPacket *avpkt)

{

    DDSContext *ctx = avctx->priv_data;

    GetByteContext *gbc = &ctx->gbc;

    AVFrame *frame = data;

    int mipmap;

    int ret;



    ff_texturedsp_init(&ctx->texdsp);

    bytestream2_init(gbc, avpkt->data, avpkt->size);



    if (bytestream2_get_bytes_left(gbc) < 128) {

        av_log(avctx, AV_LOG_ERROR, "Frame is too small (%d).\n",

               bytestream2_get_bytes_left(gbc));

        return AVERROR_INVALIDDATA;

    }



    if (bytestream2_get_le32(gbc) != MKTAG('D', 'D', 'S', ' ') ||

        bytestream2_get_le32(gbc) != 124) { // header size

        av_log(avctx, AV_LOG_ERROR, "Invalid DDS header.\n");

        return AVERROR_INVALIDDATA;

    }



    bytestream2_skip(gbc, 4); // flags



    avctx->height = bytestream2_get_le32(gbc);

    avctx->width  = bytestream2_get_le32(gbc);

    ret = av_image_check_size(avctx->width, avctx->height, 0, avctx);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Invalid image size %dx%d.\n",

               avctx->width, avctx->height);

        return ret;

    }



    /* Since codec is based on 4x4 blocks, size is aligned to 4. */

    avctx->coded_width  = FFALIGN(avctx->width,  TEXTURE_BLOCK_W);

    avctx->coded_height = FFALIGN(avctx->height, TEXTURE_BLOCK_H);



    bytestream2_skip(gbc, 4); // pitch

    bytestream2_skip(gbc, 4); // depth

    mipmap = bytestream2_get_le32(gbc);

    if (mipmap != 0)

        av_log(avctx, AV_LOG_VERBOSE, "Found %d mipmaps (ignored).\n", mipmap);



    /* Extract pixel format information, considering additional elements

     * in reserved1 and reserved2. */

    ret = parse_pixel_format(avctx);

    if (ret < 0)

        return ret;



    ret = ff_get_buffer(avctx, frame, 0);

    if (ret < 0)

        return ret;



    if (ctx->compressed) {

        int size = (avctx->coded_height / TEXTURE_BLOCK_H) *

                   (avctx->coded_width / TEXTURE_BLOCK_W) * ctx->tex_ratio;

        ctx->slice_count = av_clip(avctx->thread_count, 1,

                                   avctx->coded_height / TEXTURE_BLOCK_H);



        if (bytestream2_get_bytes_left(gbc) < size) {

            av_log(avctx, AV_LOG_ERROR,

                   "Compressed Buffer is too small (%d < %d).\n",

                   bytestream2_get_bytes_left(gbc), size);

            return AVERROR_INVALIDDATA;

        }



        /* Use the decompress function on the texture, one block per thread. */

        ctx->tex_data = gbc->buffer;

        avctx->execute2(avctx, decompress_texture_thread, frame, NULL, ctx->slice_count);

    } else if (!ctx->paletted && ctx->bpp == 4 && avctx->pix_fmt == AV_PIX_FMT_PAL8) {

        uint8_t *dst = frame->data[0];

        int x, y, i;



        /* Use the first 64 bytes as palette, then copy the rest. */

        bytestream2_get_buffer(gbc, frame->data[1], 16 * 4);

        for (i = 0; i < 16; i++) {

            AV_WN32(frame->data[1] + i*4,

                    (frame->data[1][2+i*4]<<0)+

                    (frame->data[1][1+i*4]<<8)+

                    (frame->data[1][0+i*4]<<16)+

                    (frame->data[1][3+i*4]<<24)

            );

        }

        frame->palette_has_changed = 1;



        if (bytestream2_get_bytes_left(gbc) < frame->height * frame->width / 2) {

            av_log(avctx, AV_LOG_ERROR, "Buffer is too small (%d < %d).\n",

                   bytestream2_get_bytes_left(gbc), frame->height * frame->width / 2);

            return AVERROR_INVALIDDATA;

        }



        for (y = 0; y < frame->height; y++) {

            for (x = 0; x < frame->width; x += 2) {

                uint8_t val = bytestream2_get_byte(gbc);

                dst[x    ] = val & 0xF;

                dst[x + 1] = val >> 4;

            }

            dst += frame->linesize[0];

        }

    } else {

        int linesize = av_image_get_linesize(avctx->pix_fmt, frame->width, 0);



        if (ctx->paletted) {

            int i;

            /* Use the first 1024 bytes as palette, then copy the rest. */

            bytestream2_get_buffer(gbc, frame->data[1], 256 * 4);

            for (i = 0; i < 256; i++)

                AV_WN32(frame->data[1] + i*4,

                        (frame->data[1][2+i*4]<<0)+

                        (frame->data[1][1+i*4]<<8)+

                        (frame->data[1][0+i*4]<<16)+

                        (frame->data[1][3+i*4]<<24)

                );



            frame->palette_has_changed = 1;

        }



        if (bytestream2_get_bytes_left(gbc) < frame->height * linesize) {

            av_log(avctx, AV_LOG_ERROR, "Buffer is too small (%d < %d).\n",

                   bytestream2_get_bytes_left(gbc), frame->height * linesize);

            return AVERROR_INVALIDDATA;

        }



        av_image_copy_plane(frame->data[0], frame->linesize[0],

                            gbc->buffer, linesize,

                            linesize, frame->height);

    }



    /* Run any post processing here if needed. */

    if (ctx->postproc != DDS_NONE)

        run_postproc(avctx, frame);



    /* Frame is ready to be output. */

    frame->pict_type = AV_PICTURE_TYPE_I;

    frame->key_frame = 1;

    *got_frame = 1;



    return avpkt->size;

}
