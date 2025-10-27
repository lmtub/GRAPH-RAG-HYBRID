static void do_exit(void)

{

    if (cur_stream) {

        stream_close(cur_stream);

        cur_stream = NULL;

    }

    uninit_opts();

#if CONFIG_AVFILTER

    avfilter_uninit();

#endif

    avformat_network_deinit();

    if (show_status)

        printf("\n");

    SDL_Quit();

    av_log(NULL, AV_LOG_QUIET, "");

    exit(0);

}
