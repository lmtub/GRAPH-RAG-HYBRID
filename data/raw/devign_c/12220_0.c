static inline void log_input_change(void *ctx, AVFilterLink *link, AVFilterBufferRef *ref)

{

    char old_layout_str[16], new_layout_str[16];

    av_get_channel_layout_string(old_layout_str, sizeof(old_layout_str),

                                 -1, link->channel_layout);

    av_get_channel_layout_string(new_layout_str, sizeof(new_layout_str),

                                 -1, ref->audio->channel_layout);

    av_log(ctx, AV_LOG_INFO,

           "Audio input format changed: "

           "%s:%s:%"PRId64" -> %s:%s:%u, normalizing\n",

           av_get_sample_fmt_name(link->format),

           old_layout_str, link->sample_rate,

           av_get_sample_fmt_name(ref->format),

           new_layout_str, ref->audio->sample_rate);

}
