static void opt_qmin(const char *arg)

{

    video_qmin = atoi(arg);

    if (video_qmin < 0 ||

        video_qmin > 31) {

        fprintf(stderr, "qmin must be >= 1 and <= 31\n");

        exit(1);

    }

}
