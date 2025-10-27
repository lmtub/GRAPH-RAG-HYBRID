int attribute_align_arg avcodec_decode_video2(AVCodecContext *avctx, AVFrame *picture,

                                              int *got_picture_ptr,

                                              AVPacket *avpkt)

{

    AVCodecInternal *avci = avctx->internal;

    int ret;



    *got_picture_ptr = 0;

    if ((avctx->coded_width || avctx->coded_height) && av_image_check_size(avctx->coded_width, avctx->coded_height, 0, avctx))

        return -1;



    avctx->internal->pkt = avpkt;

    ret = apply_param_change(avctx, avpkt);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error applying parameter changes.\n");

        if (avctx->err_recognition & AV_EF_EXPLODE)

            return ret;

    }



    avcodec_get_frame_defaults(picture);



    if (!avctx->refcounted_frames)

        av_frame_unref(&avci->to_free);



    if ((avctx->codec->capabilities & CODEC_CAP_DELAY) || avpkt->size || (avctx->active_thread_type & FF_THREAD_FRAME)) {

        if (HAVE_THREADS && avctx->active_thread_type & FF_THREAD_FRAME)

            ret = ff_thread_decode_frame(avctx, picture, got_picture_ptr,

                                         avpkt);

        else {

            ret = avctx->codec->decode(avctx, picture, got_picture_ptr,

                                       avpkt);

            picture->pkt_dts = avpkt->dts;

            /* get_buffer is supposed to set frame parameters */

            if (!(avctx->codec->capabilities & CODEC_CAP_DR1)) {

                picture->sample_aspect_ratio = avctx->sample_aspect_ratio;

                picture->width               = avctx->width;

                picture->height              = avctx->height;

                picture->format              = avctx->pix_fmt;

            }

        }



        emms_c(); //needed to avoid an emms_c() call before every return;



        if (ret < 0 && picture->buf[0])

            av_frame_unref(picture);



        if (*got_picture_ptr) {

            if (!avctx->refcounted_frames) {

                avci->to_free = *picture;

                avci->to_free.extended_data = avci->to_free.data;

                memset(picture->buf, 0, sizeof(picture->buf));

            }



            avctx->frame_number++;

        }

    } else

        ret = 0;



    return ret;

}
