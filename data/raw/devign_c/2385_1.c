static void opt_frame_pix_fmt(const char *arg)

{

    if (strcmp(arg, "list"))

        frame_pix_fmt = avcodec_get_pix_fmt(arg);

    else {

        list_fmts(avcodec_pix_fmt_string, PIX_FMT_NB);

        av_exit(0);

    }

}
