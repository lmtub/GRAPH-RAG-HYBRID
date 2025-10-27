static void do_video_out(AVFormatContext *s, 

                         AVOutputStream *ost, 

                         AVInputStream *ist,

                         AVPicture *in_picture,

                         int *frame_size, AVOutputStream *audio_sync)

{

    int nb_frames, i, ret;

    AVPicture *final_picture, *formatted_picture;

    AVPicture picture_format_temp, picture_crop_temp;

    static uint8_t *video_buffer;

    uint8_t *buf = NULL, *buf1 = NULL;

    AVCodecContext *enc, *dec;



#define VIDEO_BUFFER_SIZE (1024*1024)



    enc = &ost->st->codec;

    dec = &ist->st->codec;



    /* by default, we output a single frame */

    nb_frames = 1;



    *frame_size = 0;



    /* NOTE: the A/V sync is always done by considering the audio is

       the master clock. It is suffisant for transcoding or playing,

       but not for the general case */

    if (audio_sync) {

        /* compute the A-V delay and duplicate/remove frames if needed */

        double adelta, vdelta, av_delay;



        adelta = audio_sync->sync_ipts - ((double)audio_sync->sync_opts * 

            s->pts_num / s->pts_den);



        vdelta = ost->sync_ipts - ((double)ost->sync_opts *

            s->pts_num / s->pts_den);



        av_delay = adelta - vdelta;

        //            printf("delay=%f\n", av_delay);

        if (av_delay < -AV_DELAY_MAX)

            nb_frames = 2;

        else if (av_delay > AV_DELAY_MAX)

            nb_frames = 0;

    } else {

        double vdelta;



        vdelta = (double)(ost->st->pts.val) * s->pts_num / s->pts_den - (ost->sync_ipts - ost->sync_ipts_offset);

        if (vdelta < 100 && vdelta > -100 && ost->sync_ipts_offset) {

            if (vdelta < -AV_DELAY_MAX)

                nb_frames = 2;

            else if (vdelta > AV_DELAY_MAX)

                nb_frames = 0;

        } else {

            ost->sync_ipts_offset -= vdelta;

            if (!ost->sync_ipts_offset)

                ost->sync_ipts_offset = 0.000001; /* one microsecond */

        }

    }

    

#if defined(AVSYNC_DEBUG)

    static char *action[] = { "drop frame", "copy frame", "dup frame" };

    if (audio_sync)

        fprintf(stderr, "Input APTS %12.6f, output APTS %12.6f, ",

	        (double) audio_sync->sync_ipts, 

	        (double) audio_sync->st->pts.val * s->pts_num / s->pts_den);

    fprintf(stderr, "Input VPTS %12.6f, output VPTS %12.6f: %s\n",

            (double) ost->sync_ipts, 

  	    (double) ost->st->pts.val * s->pts_num / s->pts_den,

            action[nb_frames]);

#endif



    if (nb_frames <= 0) 

        return;



    if (!video_buffer)

        video_buffer = av_malloc(VIDEO_BUFFER_SIZE);

    if (!video_buffer)

        return;



    /* convert pixel format if needed */

    if (enc->pix_fmt != dec->pix_fmt) {

        int size;



        /* create temporary picture */

        size = avpicture_get_size(enc->pix_fmt, dec->width, dec->height);

        buf = av_malloc(size);

        if (!buf)

            return;

        formatted_picture = &picture_format_temp;

        avpicture_fill(formatted_picture, buf, enc->pix_fmt, dec->width, dec->height);

        

        if (img_convert(formatted_picture, enc->pix_fmt, 

                        in_picture, dec->pix_fmt, 

                        dec->width, dec->height) < 0) {

            fprintf(stderr, "pixel format conversion not handled\n");

            goto the_end;

        }

    } else {

        formatted_picture = in_picture;

    }



    /* XXX: resampling could be done before raw format convertion in

       some cases to go faster */

    /* XXX: only works for YUV420P */

    if (ost->video_resample) {

        final_picture = &ost->pict_tmp;

        img_resample(ost->img_resample_ctx, final_picture, formatted_picture);

    } else if (ost->video_crop) {

        picture_crop_temp.data[0] = formatted_picture->data[0] +

                (ost->topBand * formatted_picture->linesize[0]) + ost->leftBand;



        picture_crop_temp.data[1] = formatted_picture->data[1] +

                ((ost->topBand >> 1) * formatted_picture->linesize[1]) +

                (ost->leftBand >> 1);



        picture_crop_temp.data[2] = formatted_picture->data[2] +

                ((ost->topBand >> 1) * formatted_picture->linesize[2]) +

                (ost->leftBand >> 1);



        picture_crop_temp.linesize[0] = formatted_picture->linesize[0];

        picture_crop_temp.linesize[1] = formatted_picture->linesize[1];

        picture_crop_temp.linesize[2] = formatted_picture->linesize[2];

        final_picture = &picture_crop_temp;

    } else {

        final_picture = formatted_picture;

    }

    /* duplicates frame if needed */

    /* XXX: pb because no interleaving */

    for(i=0;i<nb_frames;i++) {

        if (s->oformat->flags & AVFMT_RAWPICTURE) {

            /* raw pictures are written as AVPicture structure to

               avoid any copies. We support temorarily the older

               method. */

            av_write_frame(s, ost->index, 

                           (uint8_t *)final_picture, sizeof(AVPicture));

        } else {

            AVFrame big_picture;

            

            memset(&big_picture, 0, sizeof(AVFrame));

            *(AVPicture*)&big_picture= *final_picture;

                        

            /* handles sameq here. This is not correct because it may

               not be a global option */

            if (same_quality) {

                big_picture.quality = ist->st->quality;

            }else

                big_picture.quality = ost->st->quality;

            

            ret = avcodec_encode_video(enc, 

                                       video_buffer, VIDEO_BUFFER_SIZE,

                                       &big_picture);

            //enc->frame_number = enc->real_pict_num;

            av_write_frame(s, ost->index, video_buffer, ret);

            *frame_size = ret;

            //fprintf(stderr,"\nFrame: %3d %3d size: %5d type: %d",

            //        enc->frame_number-1, enc->real_pict_num, ret,

            //        enc->pict_type);

            /* if two pass, output log */

            if (ost->logfile && enc->stats_out) {

                fprintf(ost->logfile, "%s", enc->stats_out);

            }

        }

        ost->frame_number++;

    }

 the_end:

    av_free(buf);

    av_free(buf1);

}
