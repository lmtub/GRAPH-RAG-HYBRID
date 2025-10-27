av_cold int ffv1_init_slice_contexts(FFV1Context *f)
{
    int i;
    f->slice_count = f->num_h_slices * f->num_v_slices;
    av_assert0(f->slice_count > 0);
    for (i = 0; i < f->slice_count; i++) {
        FFV1Context *fs = av_mallocz(sizeof(*fs));
        int sx          = i % f->num_h_slices;
        int sy          = i / f->num_h_slices;
        int sxs         = f->avctx->width  *  sx      / f->num_h_slices;
        int sxe         = f->avctx->width  * (sx + 1) / f->num_h_slices;
        int sys         = f->avctx->height *  sy      / f->num_v_slices;
        int sye         = f->avctx->height * (sy + 1) / f->num_v_slices;
        f->slice_context[i] = fs;
        memcpy(fs, f, sizeof(*fs));
        memset(fs->rc_stat2, 0, sizeof(fs->rc_stat2));
        fs->slice_width  = sxe - sxs;
        fs->slice_height = sye - sys;
        fs->slice_x      = sxs;
        fs->slice_y      = sys;
        fs->sample_buffer = av_malloc(3 * MAX_PLANES * (fs->width + 6) *
                                      sizeof(*fs->sample_buffer));
        if (!fs->sample_buffer)
    }
    return 0;
}