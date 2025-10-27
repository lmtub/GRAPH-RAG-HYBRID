static void ff_dlog_link(void *ctx, AVFilterLink *link, int end)

{

    if (link->type == AVMEDIA_TYPE_VIDEO) {

        av_dlog(ctx,

                "link[%p s:%dx%d fmt:%-16s %-16s->%-16s]%s",

                link, link->w, link->h,

                av_pix_fmt_descriptors[link->format].name,

                link->src ? link->src->filter->name : "",

                link->dst ? link->dst->filter->name : "",

                end ? "\n" : "");

    } else {

        char buf[128];

        av_get_channel_layout_string(buf, sizeof(buf), -1, link->channel_layout);



        av_dlog(ctx,

                "link[%p r:%"PRId64" cl:%s fmt:%-16s %-16s->%-16s]%s",

                link, link->sample_rate, buf,

                av_get_sample_fmt_name(link->format),

                link->src ? link->src->filter->name : "",

                link->dst ? link->dst->filter->name : "",

                end ? "\n" : "");

    }

}
