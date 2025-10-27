static void opt_qmax(const char *arg)

{

    video_qmax = atoi(arg);

    if (video_qmax < 0 ||

        video_qmax > 31) {

        fprintf(stderr, "qmax must be >= 1 and <= 31\n");

        exit(1);

    }

}
