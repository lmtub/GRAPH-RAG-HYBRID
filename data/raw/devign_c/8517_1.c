static int mpeg_decode_slice(AVCodecContext *avctx, 

                              AVPicture *pict,

                              int start_code,

                              UINT8 *buf, int buf_size)

{

    Mpeg1Context *s1 = avctx->priv_data;

    MpegEncContext *s = &s1->mpeg_enc_ctx;

    int ret;



    start_code = (start_code - 1) & 0xff;

    if (start_code >= s->mb_height)

        return -1;

    s->last_dc[0] = 1 << (7 + s->intra_dc_precision);

    s->last_dc[1] = s->last_dc[0];

    s->last_dc[2] = s->last_dc[0];

    memset(s->last_mv, 0, sizeof(s->last_mv));

    s->mb_x = -1;

    s->mb_y = start_code;

    s->mb_incr = 0;



    /* start frame decoding */

    if (s->first_slice) {

        s->first_slice = 0;

        MPV_frame_start(s);

    }



    init_get_bits(&s->gb, buf, buf_size);



    s->qscale = get_qscale(s);

    /* extra slice info */

    while (get_bits1(&s->gb) != 0) {

        skip_bits(&s->gb, 8);

    }



    for(;;) {

        clear_blocks(s->block[0]);

        emms_c();

        ret = mpeg_decode_mb(s, s->block);

        dprintf("ret=%d\n", ret);

        if (ret < 0)

            return -1;

        if (ret == 1)

            break;

        MPV_decode_mb(s, s->block);

    }

    emms_c();



    /* end of slice reached */

    if (s->mb_x == (s->mb_width - 1) &&

        s->mb_y == (s->mb_height - 1)) {

        /* end of image */

        UINT8 **picture;



        MPV_frame_end(s);



        /* XXX: incorrect reported qscale for mpeg2 */

        if (s->pict_type == B_TYPE) {

            picture = s->current_picture;

            avctx->quality = s->qscale;

        } else {

            /* latency of 1 frame for I and P frames */

            /* XXX: use another variable than picture_number */

            if (s->picture_number == 0) {

                picture = NULL;

            } else {

                picture = s->last_picture;

                avctx->quality = s->last_qscale;

            }

            s->last_qscale = s->qscale;

            s->picture_number++;

        }

        if (picture) {

            pict->data[0] = picture[0];

            pict->data[1] = picture[1];

            pict->data[2] = picture[2];

            pict->linesize[0] = s->linesize;

            pict->linesize[1] = s->linesize / 2;

            pict->linesize[2] = s->linesize / 2;

            return 1;

        } else {

            return 0;

        }

    } else {

        return 0;

    }

}
