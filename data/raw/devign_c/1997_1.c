int attribute_align_arg avcodec_decode_video2(AVCodecContext *avctx, AVFrame *picture,

                         int *got_picture_ptr,

                         AVPacket *avpkt)

{

    int ret;



    *got_picture_ptr= 0;

    if((avctx->coded_width||avctx->coded_height) && av_image_check_size(avctx->coded_width, avctx->coded_height, 0, avctx))

        return -1;



    if((avctx->codec->capabilities & CODEC_CAP_DELAY) || avpkt->size || (avctx->active_thread_type&FF_THREAD_FRAME)){

        av_packet_split_side_data(avpkt);

        apply_param_change(avctx, avpkt);

        avctx->pkt = avpkt;

        if (HAVE_THREADS && avctx->active_thread_type&FF_THREAD_FRAME)

             ret = ff_thread_decode_frame(avctx, picture, got_picture_ptr,

                                          avpkt);

        else {

            ret = avctx->codec->decode(avctx, picture, got_picture_ptr,

                              avpkt);

            picture->pkt_dts= avpkt->dts;



            if(!avctx->has_b_frames){

            picture->pkt_pos= avpkt->pos;

            }

            //FIXME these should be under if(!avctx->has_b_frames)

            if (!picture->sample_aspect_ratio.num)

                picture->sample_aspect_ratio = avctx->sample_aspect_ratio;

            if (!picture->width)

                picture->width = avctx->width;

            if (!picture->height)

                picture->height = avctx->height;

            if (picture->format == PIX_FMT_NONE)

                picture->format = avctx->pix_fmt;

        }



        emms_c(); //needed to avoid an emms_c() call before every return;





        if (*got_picture_ptr){

            avctx->frame_number++;

            picture->best_effort_timestamp = guess_correct_pts(avctx,

                                                            picture->pkt_pts,

                                                            picture->pkt_dts);

        }

    }else

        ret= 0;



    return ret;

}
