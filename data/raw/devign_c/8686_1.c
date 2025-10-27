int ff_raw_read_header(AVFormatContext *s, AVFormatParameters *ap)
{
    AVStream *st;
    enum CodecID id;
    st = avformat_new_stream(s, NULL);
    if (!st)
        return AVERROR(ENOMEM);
        id = s->iformat->value;
        if (id == CODEC_ID_RAWVIDEO) {
            st->codec->codec_type = AVMEDIA_TYPE_VIDEO;
        } else {
            st->codec->codec_type = AVMEDIA_TYPE_AUDIO;
        st->codec->codec_id = id;
        switch(st->codec->codec_type) {
        case AVMEDIA_TYPE_AUDIO: {
            RawAudioDemuxerContext *s1 = s->priv_data;
            st->codec->channels = 1;
            if (id == CODEC_ID_ADPCM_G722)
                st->codec->sample_rate = 16000;
            if (s1 && s1->sample_rate)
                st->codec->sample_rate = s1->sample_rate;
            if (s1 && s1->channels)
                st->codec->channels    = s1->channels;
            st->codec->bits_per_coded_sample = av_get_bits_per_sample(st->codec->codec_id);
            assert(st->codec->bits_per_coded_sample > 0);
            st->codec->block_align = st->codec->bits_per_coded_sample*st->codec->channels/8;
            av_set_pts_info(st, 64, 1, st->codec->sample_rate);
            break;
        case AVMEDIA_TYPE_VIDEO: {
            FFRawVideoDemuxerContext *s1 = s->priv_data;
            int width = 0, height = 0, ret = 0;
            enum PixelFormat pix_fmt;
            AVRational framerate;
            if (s1->video_size && (ret = av_parse_video_size(&width, &height, s1->video_size)) < 0) {
                av_log(s, AV_LOG_ERROR, "Couldn't parse video size.\n");
                goto fail;
            if ((pix_fmt = av_get_pix_fmt(s1->pixel_format)) == PIX_FMT_NONE) {
                av_log(s, AV_LOG_ERROR, "No such pixel format: %s.\n", s1->pixel_format);
                ret = AVERROR(EINVAL);
                goto fail;
            if ((ret = av_parse_video_rate(&framerate, s1->framerate)) < 0) {
                av_log(s, AV_LOG_ERROR, "Could not parse framerate: %s.\n", s1->framerate);
                goto fail;
            av_set_pts_info(st, 64, framerate.den, framerate.num);
            st->codec->width  = width;
            st->codec->height = height;
            st->codec->pix_fmt = pix_fmt;
fail:
            return ret;
        default:
            return -1;
    return 0;