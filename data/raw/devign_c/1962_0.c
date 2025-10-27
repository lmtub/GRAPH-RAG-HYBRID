static av_cold int source_init(AVFilterContext *ctx, const char *args, void *opaque)

{

    Frei0rContext *frei0r = ctx->priv;

    char dl_name[1024], c;

    char frame_size[128] = "";

    char frame_rate[128] = "";

    AVRational frame_rate_q;



    memset(frei0r->params, 0, sizeof(frei0r->params));



    if (args)

        sscanf(args, "%127[^:]:%127[^:]:%1023[^:=]%c%255c",

               frame_size, frame_rate, dl_name, &c, frei0r->params);



    if (av_parse_video_size(&frei0r->w, &frei0r->h, frame_size) < 0) {

        av_log(ctx, AV_LOG_ERROR, "Invalid frame size: '%s'\n", frame_size);

        return AVERROR(EINVAL);

    }



    if (av_parse_video_rate(&frame_rate_q, frame_rate) < 0 ||

        frame_rate_q.den <= 0 || frame_rate_q.num <= 0) {

        av_log(ctx, AV_LOG_ERROR, "Invalid frame rate: '%s'\n", frame_rate);

        return AVERROR(EINVAL);

    }

    frei0r->time_base.num = frame_rate_q.den;

    frei0r->time_base.den = frame_rate_q.num;



    return frei0r_init(ctx, dl_name, F0R_PLUGIN_TYPE_SOURCE);

}
