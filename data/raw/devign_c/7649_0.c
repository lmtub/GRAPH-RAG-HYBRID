static void end_last_frame(AVFilterContext *ctx)

{

    TileContext *tile    = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];

    AVFilterBufferRef *out_buf = outlink->out_buf;



    outlink->out_buf = NULL;

    ff_start_frame(outlink, out_buf);

    while (tile->current < tile->nb_frames)

        draw_blank_frame(ctx, out_buf);

    ff_draw_slice(outlink, 0, out_buf->video->h, 1);

    ff_end_frame(outlink);

    tile->current = 0;

}
