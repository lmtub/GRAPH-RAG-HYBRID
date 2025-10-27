void avfilter_start_frame(AVFilterLink *link, AVFilterBufferRef *picref)

{

    void (*start_frame)(AVFilterLink *, AVFilterBufferRef *);

    AVFilterPad *dst = &link_dpad(link);



    FF_DPRINTF_START(NULL, start_frame); ff_dprintf_link(NULL, link, 0); dprintf(NULL, " "); ff_dprintf_ref(NULL, picref, 1);



    if (!(start_frame = dst->start_frame))

        start_frame = avfilter_default_start_frame;



    /* prepare to copy the picture if it has insufficient permissions */

    if ((dst->min_perms & picref->perms) != dst->min_perms ||

         dst->rej_perms & picref->perms) {

        av_log(link->dst, AV_LOG_DEBUG,

                "frame copy needed (have perms %x, need %x, reject %x)\n",

                picref->perms,

                link_dpad(link).min_perms, link_dpad(link).rej_perms);



        link->cur_buf = avfilter_default_get_video_buffer(link, dst->min_perms, link->w, link->h);

        link->src_buf = picref;

        avfilter_copy_buffer_ref_props(link->cur_buf, link->src_buf);

    }

    else

        link->cur_buf = picref;



    start_frame(link, link->cur_buf);

}
