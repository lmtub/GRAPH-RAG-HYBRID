static int yop_read_header(AVFormatContext *s)

{

    YopDecContext *yop = s->priv_data;

    AVIOContext *pb  = s->pb;



    AVCodecContext *audio_dec, *video_dec;

    AVStream *audio_stream, *video_stream;



    int frame_rate, ret;



    audio_stream = avformat_new_stream(s, NULL);

    video_stream = avformat_new_stream(s, NULL);





    // Extra data that will be passed to the decoder

    video_stream->codec->extradata_size = 8;



    video_stream->codec->extradata = av_mallocz(video_stream->codec->extradata_size +

                                                FF_INPUT_BUFFER_PADDING_SIZE);



    if (!video_stream->codec->extradata)




    // Audio

    audio_dec               = audio_stream->codec;

    audio_dec->codec_type   = AVMEDIA_TYPE_AUDIO;

    audio_dec->codec_id     = AV_CODEC_ID_ADPCM_IMA_APC;

    audio_dec->channels     = 1;

    audio_dec->sample_rate  = 22050;



    // Video

    video_dec               = video_stream->codec;

    video_dec->codec_type   = AVMEDIA_TYPE_VIDEO;

    video_dec->codec_id     = AV_CODEC_ID_YOP;



    avio_skip(pb, 6);



    frame_rate              = avio_r8(pb);

    yop->frame_size         = avio_r8(pb) * 2048;

    video_dec->width        = avio_rl16(pb);

    video_dec->height       = avio_rl16(pb);



    video_stream->sample_aspect_ratio = (AVRational){1, 2};



    ret = avio_read(pb, video_dec->extradata, 8);

    if (ret < 8)

        return ret < 0 ? ret : AVERROR_EOF;



    yop->palette_size       = video_dec->extradata[0] * 3 + 4;

    yop->audio_block_length = AV_RL16(video_dec->extradata + 6);



    // 1840 samples per frame, 1 nibble per sample; hence 1840/2 = 920

    if (yop->audio_block_length < 920 ||

        yop->audio_block_length + yop->palette_size >= yop->frame_size) {

        av_log(s, AV_LOG_ERROR, "YOP has invalid header\n");

        return AVERROR_INVALIDDATA;

    }



    avio_seek(pb, 2048, SEEK_SET);



    avpriv_set_pts_info(video_stream, 32, 1, frame_rate);



    return 0;

}