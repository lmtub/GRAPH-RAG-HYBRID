av_cold int ff_vc1_decode_init_alloc_tables(VC1Context *v)

{

    MpegEncContext *s = &v->s;

    int i;

    int mb_height = FFALIGN(s->mb_height, 2);



    /* Allocate mb bitplanes */

    v->mv_type_mb_plane = av_malloc (s->mb_stride * mb_height);

    v->direct_mb_plane  = av_malloc (s->mb_stride * mb_height);

    v->forward_mb_plane = av_malloc (s->mb_stride * mb_height);

    v->fieldtx_plane    = av_mallocz(s->mb_stride * mb_height);

    v->acpred_plane     = av_malloc (s->mb_stride * mb_height);

    v->over_flags_plane = av_malloc (s->mb_stride * mb_height);

    if (!v->mv_type_mb_plane || !v->direct_mb_plane || !v->forward_mb_plane ||

        !v->fieldtx_plane || !v->acpred_plane || !v->over_flags_plane)

        goto error;



    v->n_allocated_blks = s->mb_width + 2;

    v->block            = av_malloc(sizeof(*v->block) * v->n_allocated_blks);

    v->cbp_base         = av_malloc(sizeof(v->cbp_base[0]) * 2 * s->mb_stride);

    if (!v->block || !v->cbp_base)

        goto error;

    v->cbp              = v->cbp_base + s->mb_stride;

    v->ttblk_base       = av_malloc(sizeof(v->ttblk_base[0]) * 2 * s->mb_stride);

    if (!v->ttblk_base)

        goto error;

    v->ttblk            = v->ttblk_base + s->mb_stride;

    v->is_intra_base    = av_mallocz(sizeof(v->is_intra_base[0]) * 2 * s->mb_stride);

    if (!v->is_intra_base)

        goto error;

    v->is_intra         = v->is_intra_base + s->mb_stride;

    v->luma_mv_base     = av_malloc(sizeof(v->luma_mv_base[0]) * 2 * s->mb_stride);

    if (!v->luma_mv_base)

        goto error;

    v->luma_mv          = v->luma_mv_base + s->mb_stride;



    /* allocate block type info in that way so it could be used with s->block_index[] */

    v->mb_type_base = av_malloc(s->b8_stride * (mb_height * 2 + 1) + s->mb_stride * (mb_height + 1) * 2);

    if (!v->mb_type_base)

        goto error;

    v->mb_type[0]   = v->mb_type_base + s->b8_stride + 1;

    v->mb_type[1]   = v->mb_type_base + s->b8_stride * (mb_height * 2 + 1) + s->mb_stride + 1;

    v->mb_type[2]   = v->mb_type[1] + s->mb_stride * (mb_height + 1);



    /* allocate memory to store block level MV info */

    v->blk_mv_type_base = av_mallocz(     s->b8_stride * (mb_height * 2 + 1) + s->mb_stride * (mb_height + 1) * 2);

    if (!v->blk_mv_type_base)

        goto error;

    v->blk_mv_type      = v->blk_mv_type_base + s->b8_stride + 1;

    v->mv_f_base        = av_mallocz(2 * (s->b8_stride * (mb_height * 2 + 1) + s->mb_stride * (mb_height + 1) * 2));

    if (!v->mv_f_base)

        goto error;

    v->mv_f[0]          = v->mv_f_base + s->b8_stride + 1;

    v->mv_f[1]          = v->mv_f[0] + (s->b8_stride * (mb_height * 2 + 1) + s->mb_stride * (mb_height + 1) * 2);

    v->mv_f_next_base   = av_mallocz(2 * (s->b8_stride * (mb_height * 2 + 1) + s->mb_stride * (mb_height + 1) * 2));

    if (!v->mv_f_next_base)

        goto error;

    v->mv_f_next[0]     = v->mv_f_next_base + s->b8_stride + 1;

    v->mv_f_next[1]     = v->mv_f_next[0] + (s->b8_stride * (mb_height * 2 + 1) + s->mb_stride * (mb_height + 1) * 2);



    ff_intrax8_common_init(&v->x8,s);



    if (s->avctx->codec_id == AV_CODEC_ID_WMV3IMAGE || s->avctx->codec_id == AV_CODEC_ID_VC1IMAGE) {

        for (i = 0; i < 4; i++) {

            v->sr_rows[i >> 1][i & 1] = av_malloc(v->output_width);

            if (!v->sr_rows[i >> 1][i & 1])

                goto error;

        }

    }



    return 0;



error:

    ff_vc1_decode_end(s->avctx);

    return AVERROR(ENOMEM);

}
