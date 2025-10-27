static void do_video_out(AVFormatContext *s, 

                         AVOutputStream *ost, 

                         AVInputStream *ist,

                         AVPicture *picture1)

{

    int n1, n2, nb, i, ret, frame_number;

    AVPicture *picture, *picture2, *pict;

    AVPicture picture_tmp1, picture_tmp2;

    UINT8 video_buffer[128*1024];

    UINT8 *buf = NULL, *buf1 = NULL;

    AVCodecContext *enc, *dec;



    enc = &ost->st->codec;

    dec = &ist->st->codec;



    frame_number = ist->frame_number;

    /* first drop frame if needed */

    n1 = ((INT64)frame_number * enc->frame_rate) / dec->frame_rate;

    n2 = (((INT64)frame_number + 1) * enc->frame_rate) / dec->frame_rate;

    nb = n2 - n1;

    if (nb <= 0)

        return;

    

    /* deinterlace : must be done before any resize */

    if (do_deinterlace) {

        int size;



        /* create temporary picture */

        size = avpicture_get_size(dec->pix_fmt, dec->width, dec->height);

        buf1 = malloc(size);

        if (!buf1)

            return;

        

        picture2 = &picture_tmp2;

        avpicture_fill(picture2, buf1, dec->pix_fmt, dec->width, dec->height);



        if (avpicture_deinterlace(picture2, picture1, 

                                  dec->pix_fmt, dec->width, dec->height) < 0) {

            /* if error, do not deinterlace */

            free(buf1);

            buf1 = NULL;

            picture2 = picture1;

        }

    } else {

        picture2 = picture1;

    }



    /* convert pixel format if needed */

    if (enc->pix_fmt != dec->pix_fmt) {

        int size;



        /* create temporary picture */

        size = avpicture_get_size(enc->pix_fmt, dec->width, dec->height);

        buf = malloc(size);

        if (!buf)

            return;

        pict = &picture_tmp1;

        avpicture_fill(pict, buf, enc->pix_fmt, dec->width, dec->height);

        

        if (img_convert(pict, enc->pix_fmt, 

                        picture2, dec->pix_fmt, 

                        dec->width, dec->height) < 0) {

            fprintf(stderr, "pixel format conversion not handled\n");

            goto the_end;

        }

    } else {

        pict = picture2;

    }



    /* XXX: resampling could be done before raw format convertion in

       some cases to go faster */

    /* XXX: only works for YUV420P */

    if (ost->video_resample) {

        picture = &ost->pict_tmp;

        img_resample(ost->img_resample_ctx, picture, pict);

    } else {

        picture = pict;

    }



    /* duplicates frame if needed */

    /* XXX: pb because no interleaving */

    for(i=0;i<nb;i++) {

        if (enc->codec_id != CODEC_ID_RAWVIDEO) {

            /* handles sameq here. This is not correct because it may

               not be a global option */

            if (same_quality) {

                enc->quality = dec->quality;

            }

            ret = avcodec_encode_video(enc, 

                                       video_buffer, sizeof(video_buffer), 

                                       picture);

            s->format->write_packet(s, ost->index, video_buffer, ret);

        } else {

            write_picture(s, ost->index, picture, enc->pix_fmt, enc->width, enc->height);

        }

    }

    the_end:

    if (buf)

        free(buf);

    if (buf1)

        free(buf1);

}
