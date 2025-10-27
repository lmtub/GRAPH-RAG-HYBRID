void avfilter_default_start_frame(AVFilterLink *link, AVFilterPicRef *picref)

{

    AVFilterLink *out = NULL;



    if(link->dst->output_count)

        out = link->dst->outputs[0];



    if(out) {

        out->outpic      = avfilter_get_video_buffer(out, AV_PERM_WRITE, link->w, link->h);

        out->outpic->pts = picref->pts;

        avfilter_start_frame(out, avfilter_ref_pic(out->outpic, ~0));

    }

}
