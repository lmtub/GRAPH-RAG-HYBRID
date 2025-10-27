int ff_default_query_formats(AVFilterContext *ctx)

{

    return default_query_formats_common(ctx, ff_all_channel_layouts);

}
