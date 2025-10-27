static int a64multi_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                                 const AVFrame *pict, int *got_packet)

{

    A64Context *c = avctx->priv_data;

    AVFrame *const p = avctx->coded_frame;



    int frame;

    int x, y;

    int b_height;

    int b_width;



    int req_size, ret;

    uint8_t *buf;



    int *charmap     = c->mc_charmap;

    uint8_t *colram  = c->mc_colram;

    uint8_t *charset = c->mc_charset;

    int *meta        = c->mc_meta_charset;

    int *best_cb     = c->mc_best_cb;



    int charset_size = 0x800 * (INTERLACED + 1);

    int colram_size  = 0x100 * c->mc_use_5col;

    int screen_size;



    if(CROP_SCREENS) {

        b_height = FFMIN(avctx->height,C64YRES) >> 3;

        b_width  = FFMIN(avctx->width ,C64XRES) >> 3;

        screen_size = b_width * b_height;

    } else {

        b_height = C64YRES >> 3;

        b_width  = C64XRES >> 3;

        screen_size = 0x400;

    }



    /* no data, means end encoding asap */

    if (!pict) {

        /* all done, end encoding */

        if (!c->mc_lifetime) return 0;

        /* no more frames in queue, prepare to flush remaining frames */

        if (!c->mc_frame_counter) {

            c->mc_lifetime = 0;

        }

        /* still frames in queue so limit lifetime to remaining frames */

        else c->mc_lifetime = c->mc_frame_counter;

    /* still new data available */

    } else {

        /* fill up mc_meta_charset with data until lifetime exceeds */

        if (c->mc_frame_counter < c->mc_lifetime) {

            *p = *pict;

            p->pict_type = AV_PICTURE_TYPE_I;

            p->key_frame = 1;

            to_meta_with_crop(avctx, p, meta + 32000 * c->mc_frame_counter);

            c->mc_frame_counter++;

            if (c->next_pts == AV_NOPTS_VALUE)

                c->next_pts = pict->pts;

            /* lifetime is not reached so wait for next frame first */

            return 0;

        }

    }



    /* lifetime reached so now convert X frames at once */

    if (c->mc_frame_counter == c->mc_lifetime) {

        req_size = 0;

        /* any frames to encode? */

        if (c->mc_lifetime) {

            req_size = charset_size + c->mc_lifetime*(screen_size + colram_size);

            if ((ret = ff_alloc_packet(pkt, req_size)) < 0) {

                av_log(avctx, AV_LOG_ERROR, "Error getting output packet of size %d.\n", req_size);

                return ret;

            }

            buf = pkt->data;



            /* calc optimal new charset + charmaps */

            ret = ff_init_elbg(meta, 32, 1000 * c->mc_lifetime, best_cb,

                               CHARSET_CHARS, 50, charmap, &c->randctx);

            if (ret < 0)

                return ret;

            ret = ff_do_elbg(meta, 32, 1000 * c->mc_lifetime, best_cb,

                             CHARSET_CHARS, 50, charmap, &c->randctx);

            if (ret < 0)

                return ret;



            /* create colorram map and a c64 readable charset */

            render_charset(avctx, charset, colram);



            /* copy charset to buf */

            memcpy(buf, charset, charset_size);



            /* advance pointers */

            buf      += charset_size;

            charset  += charset_size;

        }



        /* write x frames to buf */

        for (frame = 0; frame < c->mc_lifetime; frame++) {

            /* copy charmap to buf. buf is uchar*, charmap is int*, so no memcpy here, sorry */

            for (y = 0; y < b_height; y++) {

                for (x = 0; x < b_width; x++) {

                    buf[y * b_width + x] = charmap[y * b_width + x];

                }

            }

            /* advance pointers */

            buf += screen_size;

            req_size += screen_size;



            /* compress and copy colram to buf */

            if (c->mc_use_5col) {

                a64_compress_colram(buf, charmap, colram);

                /* advance pointers */

                buf += colram_size;

                req_size += colram_size;

            }



            /* advance to next charmap */

            charmap += 1000;

        }



        AV_WB32(avctx->extradata + 4,  c->mc_frame_counter);

        AV_WB32(avctx->extradata + 8,  charset_size);

        AV_WB32(avctx->extradata + 12, screen_size + colram_size);



        /* reset counter */

        c->mc_frame_counter = 0;



        pkt->pts = pkt->dts = c->next_pts;

        c->next_pts         = AV_NOPTS_VALUE;



        pkt->size   = req_size;

        pkt->flags |= AV_PKT_FLAG_KEY;

        *got_packet = !!req_size;

    }

    return 0;

}
