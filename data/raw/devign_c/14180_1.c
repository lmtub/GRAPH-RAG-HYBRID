AVRational av_guess_frame_rate(AVFormatContext *format, AVStream *st, AVFrame *frame)

{

    AVRational fr = st->r_frame_rate;

    AVRational codec_fr = st->codec->framerate;

    AVRational   avg_fr = st->avg_frame_rate;



    if (avg_fr.num > 0 && avg_fr.den > 0 && fr.num > 0 && fr.den > 0 &&

        av_q2d(avg_fr) < 70 && av_q2d(fr) > 210) {

        fr = avg_fr;

    }





    if (st->codec->ticks_per_frame > 1) {

        if (   codec_fr.num > 0 && codec_fr.den > 0 && av_q2d(codec_fr) < av_q2d(fr)*0.7

            && fabs(1.0 - av_q2d(av_div_q(avg_fr, fr))) > 0.1)

            fr = codec_fr;

    }



    return fr;

}
