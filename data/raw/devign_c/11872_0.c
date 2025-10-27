int ff_query_formats_all(AVFilterContext *ctx)

{

    return default_query_formats_common(ctx, ff_all_channel_counts);

}
