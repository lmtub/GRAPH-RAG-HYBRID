static double get_video_clock(VideoState *is)

{

    if (is->paused) {

        return is->video_current_pts;

    } else {

        return is->video_current_pts + (av_gettime() - is->video_current_pts_time) / 1000000.0;

    }

}
