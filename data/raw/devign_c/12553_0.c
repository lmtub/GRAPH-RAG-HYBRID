static av_cold int color_init(AVFilterContext *ctx, const char *args, void *opaque)

{

    ColorContext *color = ctx->priv;

    char color_string[128] = "black";

    char frame_size  [128] = "320x240";

    char frame_rate  [128] = "25";

    AVRational frame_rate_q;

    char *colon = 0, *equal = 0;

    int ret = 0;



    color->class = &color_class;



    if (args) {

        colon = strchr(args, ':');

        equal = strchr(args, '=');

    }



    if (!args || (equal && (!colon || equal < colon))) {

        av_opt_set_defaults(color);

        if ((ret = av_set_options_string(color, args, "=", ":")) < 0) {

            av_log(ctx, AV_LOG_ERROR, "Error parsing options string: '%s'\n", args);

            goto end;

        }

        if (av_parse_video_rate(&frame_rate_q, color->rate_str) < 0 ||

                frame_rate_q.den <= 0 || frame_rate_q.num <= 0) {

            av_log(ctx, AV_LOG_ERROR, "Invalid frame rate: %s\n", color->rate_str);

            ret = AVERROR(EINVAL);

            goto end;

        }

        if (av_parse_color(color->color_rgba, color->color_str, -1, ctx) < 0) {

            ret = AVERROR(EINVAL);

            goto end;

        }

    } else {

        av_log(ctx, AV_LOG_WARNING, "Flat options syntax is deprecated, use key=value pairs.\n");

        sscanf(args, "%127[^:]:%127[^:]:%127s", color_string, frame_size, frame_rate);

        if (av_parse_video_size(&color->w, &color->h, frame_size) < 0) {

            av_log(ctx, AV_LOG_ERROR, "Invalid frame size: %s\n", frame_size);

            return AVERROR(EINVAL);

        }

        if (av_parse_video_rate(&frame_rate_q, frame_rate) < 0 ||

                frame_rate_q.den <= 0 || frame_rate_q.num <= 0) {

            av_log(ctx, AV_LOG_ERROR, "Invalid frame rate: %s\n", frame_rate);

            return AVERROR(EINVAL);

        }

        if (av_parse_color(color->color_rgba, color_string, -1, ctx) < 0)

            return AVERROR(EINVAL);

    }



    color->time_base.num = frame_rate_q.den;

    color->time_base.den = frame_rate_q.num;



end:

    av_opt_free(color);

    return ret;

}
