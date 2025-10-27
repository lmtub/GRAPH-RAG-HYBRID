static int audio_read_header(AVFormatContext *s1, AVFormatParameters *ap)

{

    AudioData *s = s1->priv_data;

    AVStream *st;

    int ret;



    if (!ap || ap->sample_rate <= 0 || ap->channels <= 0)

        return -1;



    st = av_new_stream(s1, 0);

    if (!st) {

        return -ENOMEM;

    }

    s->sample_rate = ap->sample_rate;

    s->channels = ap->channels;



    ret = audio_open(s, 0, ap->device);

    if (ret < 0) {

        av_free(st);

        return AVERROR_IO;

    }



    /* take real parameters */

    st->codec->codec_type = CODEC_TYPE_AUDIO;

    st->codec->codec_id = s->codec_id;

    st->codec->sample_rate = s->sample_rate;

    st->codec->channels = s->channels;



    av_set_pts_info(st, 64, 1, 1000000);  /* 64 bits pts in us */

    return 0;

}
