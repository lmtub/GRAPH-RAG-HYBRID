static int stream_component_open(PlayerState *is, int stream_index)

{

    AVFormatContext *ic = is->ic;

    AVCodecContext *avctx;

    AVCodec *codec;

    SDL_AudioSpec wanted_spec, spec;

    AVDictionary *opts;

    AVDictionaryEntry *t = NULL;

    int ret = 0;



    if (stream_index < 0 || stream_index >= ic->nb_streams)

        return -1;

    avctx = ic->streams[stream_index]->codec;



    opts = filter_codec_opts(codec_opts, avctx->codec_id, ic, ic->streams[stream_index], NULL);



    codec = avcodec_find_decoder(avctx->codec_id);

    avctx->workaround_bugs   = workaround_bugs;

    avctx->idct_algo         = idct;

    avctx->skip_frame        = skip_frame;

    avctx->skip_idct         = skip_idct;

    avctx->skip_loop_filter  = skip_loop_filter;

    avctx->error_concealment = error_concealment;



    if (fast)

        avctx->flags2 |= AV_CODEC_FLAG2_FAST;



    if (!av_dict_get(opts, "threads", NULL, 0))

        av_dict_set(&opts, "threads", "auto", 0);

    if (avctx->codec_type == AVMEDIA_TYPE_VIDEO)

        av_dict_set(&opts, "refcounted_frames", "1", 0);

    if (!codec ||

        (ret = avcodec_open2(avctx, codec, &opts)) < 0) {

        goto fail;

    }

    if ((t = av_dict_get(opts, "", NULL, AV_DICT_IGNORE_SUFFIX))) {

        av_log(NULL, AV_LOG_ERROR, "Option %s not found.\n", t->key);

        ret =  AVERROR_OPTION_NOT_FOUND;

        goto fail;

    }



    /* prepare audio output */

    if (avctx->codec_type == AVMEDIA_TYPE_AUDIO) {

        is->sdl_sample_rate = avctx->sample_rate;



        if (!avctx->channel_layout)

            avctx->channel_layout = av_get_default_channel_layout(avctx->channels);

        if (!avctx->channel_layout) {

            fprintf(stderr, "unable to guess channel layout\n");

            ret = AVERROR_INVALIDDATA;

            goto fail;

        }

        if (avctx->channels == 1)

            is->sdl_channel_layout = AV_CH_LAYOUT_MONO;

        else

            is->sdl_channel_layout = AV_CH_LAYOUT_STEREO;

        is->sdl_channels = av_get_channel_layout_nb_channels(is->sdl_channel_layout);



        wanted_spec.format = AUDIO_S16SYS;

        wanted_spec.freq = is->sdl_sample_rate;

        wanted_spec.channels = is->sdl_channels;

        wanted_spec.silence = 0;

        wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE;

        wanted_spec.callback = sdl_audio_callback;

        wanted_spec.userdata = is;

        if (SDL_OpenAudio(&wanted_spec, &spec) < 0) {

            fprintf(stderr, "SDL_OpenAudio: %s\n", SDL_GetError());

            ret = AVERROR_UNKNOWN;

            goto fail;

        }

        is->audio_hw_buf_size = spec.size;

        is->sdl_sample_fmt          = AV_SAMPLE_FMT_S16;

        is->resample_sample_fmt     = is->sdl_sample_fmt;

        is->resample_channel_layout = avctx->channel_layout;

        is->resample_sample_rate    = avctx->sample_rate;

    }



    ic->streams[stream_index]->discard = AVDISCARD_DEFAULT;

    switch (avctx->codec_type) {

    case AVMEDIA_TYPE_AUDIO:

        is->audio_stream = stream_index;

        is->audio_st = ic->streams[stream_index];

        is->audio_buf_size  = 0;

        is->audio_buf_index = 0;



        /* init averaging filter */

        is->audio_diff_avg_coef  = exp(log(0.01) / AUDIO_DIFF_AVG_NB);

        is->audio_diff_avg_count = 0;

        /* since we do not have a precise anough audio fifo fullness,

           we correct audio sync only if larger than this threshold */

        is->audio_diff_threshold = 2.0 * SDL_AUDIO_BUFFER_SIZE / avctx->sample_rate;



        memset(&is->audio_pkt, 0, sizeof(is->audio_pkt));

        packet_queue_init(&is->audioq);

        SDL_PauseAudio(0);

        break;

    case AVMEDIA_TYPE_VIDEO:

        is->video_stream = stream_index;

        is->video_st = ic->streams[stream_index];



        packet_queue_init(&is->videoq);

        is->video_tid = SDL_CreateThread(video_thread, is);

        break;

    case AVMEDIA_TYPE_SUBTITLE:

        is->subtitle_stream = stream_index;

        is->subtitle_st = ic->streams[stream_index];

        packet_queue_init(&is->subtitleq);



        is->subtitle_tid = SDL_CreateThread(subtitle_thread, is);

        break;

    default:

        break;

    }



fail:

    av_dict_free(&opts);



    return ret;

}
