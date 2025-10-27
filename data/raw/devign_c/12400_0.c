static int source_request_frame(AVFilterLink *outlink)

{

    Frei0rContext *frei0r = outlink->src->priv;

    AVFilterBufferRef *picref = ff_get_video_buffer(outlink, AV_PERM_WRITE, outlink->w, outlink->h);

    int ret;



    picref->video->pixel_aspect = (AVRational) {1, 1};

    picref->pts = frei0r->pts++;

    picref->pos = -1;



    ret = ff_start_frame(outlink, avfilter_ref_buffer(picref, ~0));

    if (ret < 0)

        goto fail;



    frei0r->update(frei0r->instance, av_rescale_q(picref->pts, frei0r->time_base, (AVRational){1,1000}),

                   NULL, (uint32_t *)picref->data[0]);

    ret = ff_draw_slice(outlink, 0, outlink->h, 1);

    if (ret < 0)

        goto fail;



    ret = ff_end_frame(outlink);



fail:

    avfilter_unref_buffer(picref);



    return ret;

}
