static int find_smallest_bounding_rectangle(AVSubtitle *s)

{

    uint8_t transp_color[256] = { 0 };

    int y1, y2, x1, x2, y, w, h, i;

    uint8_t *bitmap;



    if (s->num_rects == 0 || s->rects == NULL || s->rects[0]->w <= 0 || s->rects[0]->h <= 0)

        return 0;



    for(i = 0; i < s->rects[0]->nb_colors; i++) {

        if ((((uint32_t*)s->rects[0]->pict.data[1])[i] >> 24) == 0)

            transp_color[i] = 1;

    }

    y1 = 0;

    while (y1 < s->rects[0]->h && is_transp(s->rects[0]->pict.data[0] + y1 * s->rects[0]->pict.linesize[0],

                                  1, s->rects[0]->w, transp_color))

        y1++;

    if (y1 == s->rects[0]->h) {

        av_freep(&s->rects[0]->pict.data[0]);

        s->rects[0]->w = s->rects[0]->h = 0;

        return 0;

    }



    y2 = s->rects[0]->h - 1;

    while (y2 > 0 && is_transp(s->rects[0]->pict.data[0] + y2 * s->rects[0]->pict.linesize[0], 1,

                               s->rects[0]->w, transp_color))

        y2--;

    x1 = 0;

    while (x1 < (s->rects[0]->w - 1) && is_transp(s->rects[0]->pict.data[0] + x1, s->rects[0]->pict.linesize[0],

                                        s->rects[0]->h, transp_color))

        x1++;

    x2 = s->rects[0]->w - 1;

    while (x2 > 0 && is_transp(s->rects[0]->pict.data[0] + x2, s->rects[0]->pict.linesize[0], s->rects[0]->h,

                                  transp_color))

        x2--;

    w = x2 - x1 + 1;

    h = y2 - y1 + 1;

    bitmap = av_malloc(w * h);

    if (!bitmap)

        return 1;

    for(y = 0; y < h; y++) {

        memcpy(bitmap + w * y, s->rects[0]->pict.data[0] + x1 + (y1 + y) * s->rects[0]->pict.linesize[0], w);

    }

    av_freep(&s->rects[0]->pict.data[0]);

    s->rects[0]->pict.data[0] = bitmap;

    s->rects[0]->pict.linesize[0] = w;

    s->rects[0]->w = w;

    s->rects[0]->h = h;

    s->rects[0]->x += x1;

    s->rects[0]->y += y1;

    return 1;

}
