static void dump_stream_format(AVFormatContext *ic, int i, int index, int is_output)

{

    char buf[256];

    int flags = (is_output ? ic->oformat->flags : ic->iformat->flags);

    AVStream *st = ic->streams[i];

    int g = ff_gcd(st->time_base.num, st->time_base.den);

    avcodec_string(buf, sizeof(buf), st->codec, is_output);

    av_log(NULL, AV_LOG_INFO, "    Stream #%d.%d", index, i);

    /* the pid is an important information, so we display it */

    /* XXX: add a generic system */

    if (flags & AVFMT_SHOW_IDS)

        av_log(NULL, AV_LOG_INFO, "[0x%x]", st->id);

    if (strlen(st->language) > 0)

        av_log(NULL, AV_LOG_INFO, "(%s)", st->language);

    av_log(NULL, AV_LOG_DEBUG, ", %d/%d", st->time_base.num/g, st->time_base.den/g);

    av_log(NULL, AV_LOG_INFO, ": %s", buf);

    if(st->codec->codec_type == CODEC_TYPE_VIDEO){

        if(st->r_frame_rate.den && st->r_frame_rate.num)

            av_log(NULL, AV_LOG_INFO, ", %5.2f fps(r)", av_q2d(st->r_frame_rate));

/*      else if(st->time_base.den && st->time_base.num)

            av_log(NULL, AV_LOG_INFO, ", %5.2f fps(m)", 1/av_q2d(st->time_base));*/

        else

            av_log(NULL, AV_LOG_INFO, ", %5.2f fps(c)", 1/av_q2d(st->codec->time_base));

    }

    av_log(NULL, AV_LOG_INFO, "\n");

}
