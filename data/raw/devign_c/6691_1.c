static void stream_component_close(VideoState *is, int stream_index)

{

    AVFormatContext *ic = is->ic;

    AVCodecContext *avctx;



    if (stream_index < 0 || stream_index >= ic->nb_streams)

        return;

    avctx = ic->streams[stream_index]->codec;



    switch(avctx->codec_type) {

    case AVMEDIA_TYPE_AUDIO:

        packet_queue_abort(&is->audioq);



        SDL_CloseAudio();



        packet_queue_end(&is->audioq);

        av_free_packet(&is->audio_pkt);

        if (is->reformat_ctx)

            av_audio_convert_free(is->reformat_ctx);

        is->reformat_ctx = NULL;



        if (is->rdft) {

            av_rdft_end(is->rdft);

            av_freep(&is->rdft_data);



        }

        break;

    case AVMEDIA_TYPE_VIDEO:

        packet_queue_abort(&is->videoq);



        /* note: we also signal this mutex to make sure we deblock the

           video thread in all cases */

        SDL_LockMutex(is->pictq_mutex);

        SDL_CondSignal(is->pictq_cond);

        SDL_UnlockMutex(is->pictq_mutex);



        SDL_WaitThread(is->video_tid, NULL);



        packet_queue_end(&is->videoq);

        break;

    case AVMEDIA_TYPE_SUBTITLE:

        packet_queue_abort(&is->subtitleq);



        /* note: we also signal this mutex to make sure we deblock the

           video thread in all cases */

        SDL_LockMutex(is->subpq_mutex);

        is->subtitle_stream_changed = 1;



        SDL_CondSignal(is->subpq_cond);

        SDL_UnlockMutex(is->subpq_mutex);



        SDL_WaitThread(is->subtitle_tid, NULL);



        packet_queue_end(&is->subtitleq);

        break;

    default:

        break;

    }



    ic->streams[stream_index]->discard = AVDISCARD_ALL;

    avcodec_close(avctx);

    switch(avctx->codec_type) {

    case AVMEDIA_TYPE_AUDIO:

        is->audio_st = NULL;

        is->audio_stream = -1;

        break;

    case AVMEDIA_TYPE_VIDEO:

        is->video_st = NULL;

        is->video_stream = -1;

        break;

    case AVMEDIA_TYPE_SUBTITLE:

        is->subtitle_st = NULL;

        is->subtitle_stream = -1;

        break;

    default:

        break;

    }

}