static void compute_pkt_fields2(AVStream *st, AVPacket *pkt){

    int b_frames = FFMAX(st->codec.has_b_frames, st->codec.max_b_frames);

    int num, den, frame_size;



//    av_log(NULL, AV_LOG_DEBUG, "av_write_frame: pts:%lld dts:%lld cur_dts:%lld b:%d size:%d st:%d\n", pkt->pts, pkt->dts, st->cur_dts, b_frames, pkt->size, pkt->stream_index);

    

/*    if(pkt->pts == AV_NOPTS_VALUE && pkt->dts == AV_NOPTS_VALUE)

        return -1;*/

            

    if(pkt->pts != AV_NOPTS_VALUE)

        pkt->pts = av_rescale(pkt->pts, st->time_base.den, AV_TIME_BASE * (int64_t)st->time_base.num);

    if(pkt->dts != AV_NOPTS_VALUE)

        pkt->dts = av_rescale(pkt->dts, st->time_base.den, AV_TIME_BASE * (int64_t)st->time_base.num);



    /* duration field */

    pkt->duration = av_rescale(pkt->duration, st->time_base.den, AV_TIME_BASE * (int64_t)st->time_base.num);

    if (pkt->duration == 0) {

        compute_frame_duration(&num, &den, st, NULL, pkt);

        if (den && num) {

            pkt->duration = av_rescale(1, num * (int64_t)st->time_base.den, den * (int64_t)st->time_base.num);

        }

    }



    //XXX/FIXME this is a temporary hack until all encoders output pts

    if((pkt->pts == 0 || pkt->pts == AV_NOPTS_VALUE) && pkt->dts == AV_NOPTS_VALUE && !b_frames){

        pkt->dts=

//        pkt->pts= st->cur_dts;

        pkt->pts= st->pts.val;

    }



    //calculate dts from pts    

    if(pkt->pts != AV_NOPTS_VALUE && pkt->dts == AV_NOPTS_VALUE){

        if(b_frames){

            if(st->last_IP_pts == AV_NOPTS_VALUE){

                st->last_IP_pts= -pkt->duration;

            }

            if(st->last_IP_pts < pkt->pts){

                pkt->dts= st->last_IP_pts;

                st->last_IP_pts= pkt->pts;

            }else

                pkt->dts= pkt->pts;

        }else

            pkt->dts= pkt->pts;

    }

    

//    av_log(NULL, AV_LOG_DEBUG, "av_write_frame: pts2:%lld dts2:%lld\n", pkt->pts, pkt->dts);

    st->cur_dts= pkt->dts;

    st->pts.val= pkt->dts;



    /* update pts */

    switch (st->codec.codec_type) {

    case CODEC_TYPE_AUDIO:

        frame_size = get_audio_frame_size(&st->codec, pkt->size);



        /* HACK/FIXME, we skip the initial 0-size packets as they are most likely equal to the encoder delay,

           but it would be better if we had the real timestamps from the encoder */

        if (frame_size >= 0 && (pkt->size || st->pts.num!=st->pts.den>>1 || st->pts.val)) {

            av_frac_add(&st->pts, (int64_t)st->time_base.den * frame_size);

        }

        break;

    case CODEC_TYPE_VIDEO:

        av_frac_add(&st->pts, (int64_t)st->time_base.den * st->codec.frame_rate_base);

        break;

    default:

        break;

    }

}
