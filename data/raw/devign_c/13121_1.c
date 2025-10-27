static int stream_component_open(VideoState *is, int stream_index)

{

    AVFormatContext *ic = is->ic;

    AVCodecContext *enc;

    AVCodec *codec;

    SDL_AudioSpec wanted_spec, spec;



    if (stream_index < 0 || stream_index >= ic->nb_streams)

        return -1;

    enc = ic->streams[stream_index]->codec;



    /* prepare audio output */

    if (enc->codec_type == CODEC_TYPE_AUDIO) {

        if (enc->channels > 0) {

            enc->request_channels = FFMIN(2, enc->channels);

        } else {

            enc->request_channels = 2;

        }

    }



    codec = avcodec_find_decoder(enc->codec_id);

    enc->debug_mv = debug_mv;

    enc->debug = debug;

    enc->workaround_bugs = workaround_bugs;

    enc->lowres = lowres;

    if(lowres) enc->flags |= CODEC_FLAG_EMU_EDGE;

    enc->idct_algo= idct;

    if(fast) enc->flags2 |= CODEC_FLAG2_FAST;

    enc->skip_frame= skip_frame;

    enc->skip_idct= skip_idct;

    enc->skip_loop_filter= skip_loop_filter;

    enc->error_recognition= error_recognition;

    enc->error_concealment= error_concealment;

    avcodec_thread_init(enc, thread_count);



    set_context_opts(enc, avcodec_opts[enc->codec_type], 0);



    if (!codec ||

        avcodec_open(enc, codec) < 0)

        return -1;



    /* prepare audio output */

    if (enc->codec_type == CODEC_TYPE_AUDIO) {

        wanted_spec.freq = enc->sample_rate;

        wanted_spec.format = AUDIO_S16SYS;

        wanted_spec.channels = enc->channels;

        wanted_spec.silence = 0;

        wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE;

        wanted_spec.callback = sdl_audio_callback;

        wanted_spec.userdata = is;

        if (SDL_OpenAudio(&wanted_spec, &spec) < 0) {

            fprintf(stderr, "SDL_OpenAudio: %s\n", SDL_GetError());

            return -1;

        }

        is->audio_hw_buf_size = spec.size;

        is->audio_src_fmt= SAMPLE_FMT_S16;

    }



    ic->streams[stream_index]->discard = AVDISCARD_DEFAULT;

    switch(enc->codec_type) {

    case CODEC_TYPE_AUDIO:

        is->audio_stream = stream_index;

        is->audio_st = ic->streams[stream_index];

        is->audio_buf_size = 0;

        is->audio_buf_index = 0;



        /* init averaging filter */

        is->audio_diff_avg_coef = exp(log(0.01) / AUDIO_DIFF_AVG_NB);

        is->audio_diff_avg_count = 0;

        /* since we do not have a precise anough audio fifo fullness,

           we correct audio sync only if larger than this threshold */

        is->audio_diff_threshold = 2.0 * SDL_AUDIO_BUFFER_SIZE / enc->sample_rate;



        memset(&is->audio_pkt, 0, sizeof(is->audio_pkt));

        packet_queue_init(&is->audioq);

        SDL_PauseAudio(0);

        break;

    case CODEC_TYPE_VIDEO:

        is->video_stream = stream_index;

        is->video_st = ic->streams[stream_index];



        is->frame_last_delay = 40e-3;

        is->frame_timer = (double)av_gettime() / 1000000.0;

        is->video_current_pts_time = av_gettime();



        packet_queue_init(&is->videoq);

        is->video_tid = SDL_CreateThread(video_thread, is);

        break;

    case CODEC_TYPE_SUBTITLE:

        is->subtitle_stream = stream_index;

        is->subtitle_st = ic->streams[stream_index];

        packet_queue_init(&is->subtitleq);



        is->subtitle_tid = SDL_CreateThread(subtitle_thread, is);

        break;

    default:

        break;

    }

    return 0;

}
