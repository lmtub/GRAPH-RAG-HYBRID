static int hls_mux_init(AVFormatContext *s)

{

    HLSContext *hls = s->priv_data;

    AVFormatContext *oc;

    AVFormatContext *vtt_oc;

    int i, ret;



    ret = avformat_alloc_output_context2(&hls->avf, hls->oformat, NULL, NULL);

    if (ret < 0)

        return ret;

    oc = hls->avf;



    oc->oformat            = hls->oformat;

    oc->interrupt_callback = s->interrupt_callback;

    oc->max_delay          = s->max_delay;

    av_dict_copy(&oc->metadata, s->metadata, 0);



    if(hls->vtt_oformat) {

        ret = avformat_alloc_output_context2(&hls->vtt_avf, hls->vtt_oformat, NULL, NULL);

        if (ret < 0)

            return ret;

        vtt_oc          = hls->vtt_avf;

        vtt_oc->oformat = hls->vtt_oformat;

        av_dict_copy(&vtt_oc->metadata, s->metadata, 0);

    }



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st;

        AVFormatContext *loc;

        if (s->streams[i]->codec->codec_type == AVMEDIA_TYPE_SUBTITLE)

            loc = vtt_oc;

        else

            loc = oc;



        if (!(st = avformat_new_stream(loc, NULL)))

            return AVERROR(ENOMEM);

        avcodec_copy_context(st->codec, s->streams[i]->codec);

        st->sample_aspect_ratio = s->streams[i]->sample_aspect_ratio;

        st->time_base = s->streams[i]->time_base;

    }

    hls->start_pos = 0;



    return 0;

}
