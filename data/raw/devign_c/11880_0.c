static void do_video_out(AVFormatContext *s, 

                         AVOutputStream *ost, 

                         AVInputStream *ist,

                         AVPicture *picture1,

                         int *frame_size)

{

    int n1, n2, nb, i, ret, frame_number, dec_frame_rate;

    AVPicture *picture, *picture2, *pict;

    AVPicture picture_tmp1, picture_tmp2;

    UINT8 *video_buffer;

    UINT8 *buf = NULL, *buf1 = NULL;

    AVCodecContext *enc, *dec;



#define VIDEO_BUFFER_SIZE (1024*1024)



    enc = &ost->st->codec;

    dec = &ist->st->codec;



    frame_number = ist->frame_number;

    dec_frame_rate = ist->st->r_frame_rate;

    //    fprintf(stderr, "\n%d", dec_frame_rate);

    /* first drop frame if needed */

    n1 = ((INT64)frame_number * enc->frame_rate) / dec_frame_rate;

    n2 = (((INT64)frame_number + 1) * enc->frame_rate) / dec_frame_rate;

    nb = n2 - n1;

    if (nb <= 0) 

        return;



    video_buffer= av_malloc(VIDEO_BUFFER_SIZE);

    if(!video_buffer) return;



    /* deinterlace : must be done before any resize */

    if (do_deinterlace) {

        int size;



        /* create temporary picture */

        size = avpicture_get_size(dec->pix_fmt, dec->width, dec->height);

        buf1 = av_malloc(size);

        if (!buf1)

            return;

        

        picture2 = &picture_tmp2;

        avpicture_fill(picture2, buf1, dec->pix_fmt, dec->width, dec->height);



        if (avpicture_deinterlace(picture2, picture1, 

                                  dec->pix_fmt, dec->width, dec->height) < 0) {

            /* if error, do not deinterlace */

            av_free(buf1);

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

        buf = av_malloc(size);

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

    nb=1;

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

                                       video_buffer, VIDEO_BUFFER_SIZE,

                                       picture);

            //enc->frame_number = enc->real_pict_num;

            s->oformat->write_packet(s, ost->index, video_buffer, ret, 0);

            *frame_size = ret;

            //fprintf(stderr,"\nFrame: %3d %3d size: %5d type: %d",

            //        enc->frame_number-1, enc->real_pict_num, ret,

            //        enc->pict_type);

        } else {

            write_picture(s, ost->index, picture, enc->pix_fmt, enc->width, enc->height);

        }

    }

    the_end:

    av_free(buf);

    av_free(buf1);

    av_free(video_buffer);

}
