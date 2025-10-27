static int mpegts_set_stream_info(AVStream *st, PESContext *pes,

                                  uint32_t stream_type, uint32_t prog_reg_desc)

{

    int old_codec_type= st->codec->codec_type;

    int old_codec_id  = st->codec->codec_id;



    if (old_codec_id != AV_CODEC_ID_NONE && avcodec_is_open(st->codec)) {

        av_log(pes->stream, AV_LOG_DEBUG, "cannot set stream info, codec is open\n");

        return 0;

    }



    avpriv_set_pts_info(st, 33, 1, 90000);

    st->priv_data = pes;

    st->codec->codec_type = AVMEDIA_TYPE_DATA;

    st->codec->codec_id   = AV_CODEC_ID_NONE;

    st->need_parsing = AVSTREAM_PARSE_FULL;

    pes->st = st;

    pes->stream_type = stream_type;



    av_log(pes->stream, AV_LOG_DEBUG,

           "stream=%d stream_type=%x pid=%x prog_reg_desc=%.4s\n",

           st->index, pes->stream_type, pes->pid, (char*)&prog_reg_desc);



    st->codec->codec_tag = pes->stream_type;



    mpegts_find_stream_type(st, pes->stream_type, ISO_types);

    if ((prog_reg_desc == AV_RL32("HDMV") ||

         prog_reg_desc == AV_RL32("HDPR")) &&

        st->codec->codec_id == AV_CODEC_ID_NONE) {

        mpegts_find_stream_type(st, pes->stream_type, HDMV_types);

        if (pes->stream_type == 0x83) {

            // HDMV TrueHD streams also contain an AC3 coded version of the

            // audio track - add a second stream for this

            AVStream *sub_st;

            // priv_data cannot be shared between streams

            PESContext *sub_pes = av_malloc(sizeof(*sub_pes));

            if (!sub_pes)

                return AVERROR(ENOMEM);

            memcpy(sub_pes, pes, sizeof(*sub_pes));



            sub_st = avformat_new_stream(pes->stream, NULL);

            if (!sub_st) {

                av_free(sub_pes);

                return AVERROR(ENOMEM);

            }



            sub_st->id = pes->pid;

            avpriv_set_pts_info(sub_st, 33, 1, 90000);

            sub_st->priv_data = sub_pes;

            sub_st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

            sub_st->codec->codec_id   = AV_CODEC_ID_AC3;

            sub_st->need_parsing = AVSTREAM_PARSE_FULL;

            sub_pes->sub_st = pes->sub_st = sub_st;

        }

    }

    if (st->codec->codec_id == AV_CODEC_ID_NONE)

        mpegts_find_stream_type(st, pes->stream_type, MISC_types);

    if (st->codec->codec_id == AV_CODEC_ID_NONE){

        st->codec->codec_id  = old_codec_id;

        st->codec->codec_type= old_codec_type;

    }



    return 0;

}
