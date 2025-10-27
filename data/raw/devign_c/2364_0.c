static void pre_process_video_frame(InputStream *ist, AVPicture *picture, void **bufp)

{

    AVCodecContext *dec;

    AVPicture *picture2;

    AVPicture picture_tmp;

    uint8_t *buf = 0;



    dec = ist->st->codec;



    /* deinterlace : must be done before any resize */

    if (FF_API_DEINTERLACE && do_deinterlace) {

        int size;



        /* create temporary picture */

        size = avpicture_get_size(dec->pix_fmt, dec->width, dec->height);

        if (size < 0)

            return;

        buf  = av_malloc(size);

        if (!buf)

            return;



        picture2 = &picture_tmp;

        avpicture_fill(picture2, buf, dec->pix_fmt, dec->width, dec->height);



        if (avpicture_deinterlace(picture2, picture,

                                 dec->pix_fmt, dec->width, dec->height) < 0) {

            /* if error, do not deinterlace */

            av_log(NULL, AV_LOG_WARNING, "Deinterlacing failed\n");

            av_free(buf);

            buf = NULL;

            picture2 = picture;

        }

    } else {

        picture2 = picture;

    }



    if (picture != picture2)

        *picture = *picture2;

    *bufp = buf;

}
