static int request_frame(AVFilterLink *outlink)

{

    AVFilterBufferRef *outpicref;

    MovieContext *movie = outlink->src->priv;

    int ret;



    if (movie->is_done)

        return AVERROR_EOF;

    if ((ret = movie_get_frame(outlink)) < 0)

        return ret;



    outpicref = avfilter_ref_buffer(movie->picref, ~0);

    avfilter_start_frame(outlink, outpicref);

    avfilter_draw_slice(outlink, 0, outlink->h, 1);

    avfilter_end_frame(outlink);





    return 0;

}