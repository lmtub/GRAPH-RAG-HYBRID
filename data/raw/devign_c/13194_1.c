static int rm_write_header(AVFormatContext *s)
{
    RMMuxContext *rm = s->priv_data;
    StreamInfo *stream;
    int n;
    AVCodecContext *codec;
    for(n=0;n<s->nb_streams;n++) {
        s->streams[n]->id = n;
        codec = s->streams[n]->codec;
        stream = &rm->streams[n];
        memset(stream, 0, sizeof(StreamInfo));
        stream->num = n;
        stream->bit_rate = codec->bit_rate;
        stream->enc = codec;
        switch(codec->codec_type) {
        case AVMEDIA_TYPE_AUDIO:
            rm->audio_stream = stream;
            stream->frame_rate = (float)codec->sample_rate / (float)codec->frame_size;
            /* XXX: dummy values */
            stream->packet_max_size = 1024;
            stream->nb_packets = 0;
            stream->total_frames = stream->nb_packets;
            break;
        case AVMEDIA_TYPE_VIDEO:
            rm->video_stream = stream;
            stream->frame_rate = (float)codec->time_base.den / (float)codec->time_base.num;
            /* XXX: dummy values */
            stream->packet_max_size = 4096;
            stream->nb_packets = 0;
            stream->total_frames = stream->nb_packets;
            break;
        default:
            return -1;
    if (rv10_write_header(s, 0, 0))
        return AVERROR_INVALIDDATA;
    avio_flush(s->pb);
    return 0;