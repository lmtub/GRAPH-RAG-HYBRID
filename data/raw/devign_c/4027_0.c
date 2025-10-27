static void opt_frame_pad_left(const char *arg)

{

    frame_padleft = atoi(arg);

    if (frame_padleft < 0) {

        fprintf(stderr, "Incorrect left pad size\n");

        av_exit(1);

    }

}
