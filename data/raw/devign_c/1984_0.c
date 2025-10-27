static void opt_qscale(const char *arg)

{

    video_qscale = atof(arg);

    if (video_qscale <= 0 ||

        video_qscale > 255) {

        fprintf(stderr, "qscale must be > 0.0 and <= 255\n");

        ffmpeg_exit(1);

    }

}
