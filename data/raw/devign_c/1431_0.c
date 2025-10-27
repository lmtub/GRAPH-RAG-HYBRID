static AVFrame *get_palette_frame(AVFilterContext *ctx)

{

    AVFrame *out;

    PaletteGenContext *s = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];

    double ratio;

    int box_id = 0;

    struct range_box *box;



    /* reference only the used colors from histogram */

    s->refs = load_color_refs(s->histogram, s->nb_refs);

    if (!s->refs) {

        av_log(ctx, AV_LOG_ERROR, "Unable to allocate references for %d different colors\n", s->nb_refs);

        return NULL;

    }



    /* create the palette frame */

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);

    if (!out)

        return NULL;

    out->pts = 0;



    /* set first box for 0..nb_refs */

    box = &s->boxes[box_id];

    box->len = s->nb_refs;

    box->sorted_by = -1;

    box->color = get_avg_color(s->refs, box);

    box->variance = -1;

    s->nb_boxes = 1;



    while (box && box->len > 1) {

        int i, rr, gr, br, longest;

        uint64_t median, box_weight = 0;



        /* compute the box weight (sum all the weights of the colors in the

         * range) and its boundings */

        uint8_t min[3] = {0xff, 0xff, 0xff};

        uint8_t max[3] = {0x00, 0x00, 0x00};

        for (i = box->start; i < box->start + box->len; i++) {

            const struct color_ref *ref = s->refs[i];

            const uint32_t rgb = ref->color;

            const uint8_t r = rgb >> 16 & 0xff, g = rgb >> 8 & 0xff, b = rgb & 0xff;

            min[0] = FFMIN(r, min[0]), max[0] = FFMAX(r, max[0]);

            min[1] = FFMIN(g, min[1]), max[1] = FFMAX(g, max[1]);

            min[2] = FFMIN(b, min[2]), max[2] = FFMAX(b, max[2]);

            box_weight += ref->count;

        }



        /* define the axis to sort by according to the widest range of colors */

        rr = max[0] - min[0];

        gr = max[1] - min[1];

        br = max[2] - min[2];

        longest = 1; // pick green by default (the color the eye is the most sensitive to)

        if (br >= rr && br >= gr) longest = 2;

        if (rr >= gr && rr >= br) longest = 0;

        if (gr >= rr && gr >= br) longest = 1; // prefer green again



        av_dlog(ctx, "box #%02X [%6d..%-6d] (%6d) w:%-6"PRIu64" ranges:[%2x %2x %2x] sort by %c (already sorted:%c) ",

                box_id, box->start, box->start + box->len - 1, box->len, box_weight,

                rr, gr, br, "rgb"[longest], box->sorted_by == longest ? 'y':'n');



        /* sort the range by its longest axis if it's not already sorted */

        if (box->sorted_by != longest) {

            cmp_func cmpf = cmp_funcs[longest];

            AV_QSORT(&s->refs[box->start], box->len, const struct color_ref *, cmpf);

            box->sorted_by = longest;

        }



        /* locate the median where to split */

        median = (box_weight + 1) >> 1;

        box_weight = 0;

        /* if you have 2 boxes, the maximum is actually #0: you must have at

         * least 1 color on each side of the split, hence the -2 */

        for (i = box->start; i < box->start + box->len - 2; i++) {

            box_weight += s->refs[i]->count;

            if (box_weight > median)

                break;

        }

        av_dlog(ctx, "split @ i=%-6d with w=%-6"PRIu64" (target=%6"PRIu64")\n", i, box_weight, median);

        split_box(s, box, i);



        box_id = get_next_box_id_to_split(s);

        box = box_id >= 0 ? &s->boxes[box_id] : NULL;

    }



    ratio = set_colorquant_ratio_meta(out, s->nb_boxes, s->nb_refs);

    av_log(ctx, AV_LOG_INFO, "%d%s colors generated out of %d colors; ratio=%f\n",

           s->nb_boxes, s->reserve_transparent ? "(+1)" : "", s->nb_refs, ratio);



    qsort(s->boxes, s->nb_boxes, sizeof(*s->boxes), cmp_color);



    write_palette(ctx, out);



    return out;

}
