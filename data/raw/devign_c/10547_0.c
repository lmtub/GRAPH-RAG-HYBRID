static int get_pcm(HEVCContext *s, int x, int y)

{

    int log2_min_pu_size = s->sps->log2_min_pu_size;

    int x_pu             = x >> log2_min_pu_size;

    int y_pu             = y >> log2_min_pu_size;



    if (x < 0 || x_pu >= s->sps->min_pu_width ||

        y < 0 || y_pu >= s->sps->min_pu_height)

        return 2;

    return s->is_pcm[y_pu * s->sps->min_pu_width + x_pu];

}
