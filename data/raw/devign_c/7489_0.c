static void opt_mb_qmin(const char *arg)

{

    video_mb_qmin = atoi(arg);

    if (video_mb_qmin < 0 ||

        video_mb_qmin > 31) {

        fprintf(stderr, "qmin must be >= 1 and <= 31\n");

        exit(1);

    }

}
