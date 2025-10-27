static void free_geotags(TiffContext *const s)

{

    int i;

    for (i = 0; i < s->geotag_count; i++) {

        if (s->geotags[i].val)

            av_freep(&s->geotags[i].val);

    }

    av_freep(&s->geotags);


}