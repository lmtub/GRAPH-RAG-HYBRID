static int write_header(AVFormatContext *s)

{

    AVCodecContext *codec = s->streams[0]->codec;



    if (s->nb_streams > 1) {

        av_log(s, AV_LOG_ERROR, "only one stream is supported\n");

        return AVERROR(EINVAL);

    }

    if (codec->codec_id != AV_CODEC_ID_WAVPACK) {

        av_log(s, AV_LOG_ERROR, "unsupported codec\n");

        return AVERROR(EINVAL);

    }

    if (codec->extradata_size > 0) {

        avpriv_report_missing_feature(s, "remuxing from matroska container");

        return AVERROR_PATCHWELCOME;

    }

    avpriv_set_pts_info(s->streams[0], 64, 1, codec->sample_rate);



    return 0;

}
