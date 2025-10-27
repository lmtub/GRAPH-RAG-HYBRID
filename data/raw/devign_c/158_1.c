static int update_size(AVCodecContext *ctx, int w, int h)

{

    VP9Context *s = ctx->priv_data;

    uint8_t *p;



    if (s->above_partition_ctx && w == ctx->width && h == ctx->height)

        return 0;



    ctx->width  = w;

    ctx->height = h;

    s->sb_cols  = (w + 63) >> 6;

    s->sb_rows  = (h + 63) >> 6;

    s->cols     = (w + 7) >> 3;

    s->rows     = (h + 7) >> 3;



#define assign(var, type, n) var = (type) p; p += s->sb_cols * n * sizeof(*var)

    av_free(s->above_partition_ctx);

    p = av_malloc(s->sb_cols * (240 + sizeof(*s->lflvl) + 16 * sizeof(*s->above_mv_ctx) +

                                64 * s->sb_rows * (1 + sizeof(*s->mv[0]) * 2)));

    if (!p)

        return AVERROR(ENOMEM);

    assign(s->above_partition_ctx, uint8_t *,              8);

    assign(s->above_skip_ctx,      uint8_t *,              8);

    assign(s->above_txfm_ctx,      uint8_t *,              8);

    assign(s->above_mode_ctx,      uint8_t *,             16);

    assign(s->above_y_nnz_ctx,     uint8_t *,             16);

    assign(s->above_uv_nnz_ctx[0], uint8_t *,              8);

    assign(s->above_uv_nnz_ctx[1], uint8_t *,              8);

    assign(s->intra_pred_data[0],  uint8_t *,             64);

    assign(s->intra_pred_data[1],  uint8_t *,             32);

    assign(s->intra_pred_data[2],  uint8_t *,             32);

    assign(s->above_segpred_ctx,   uint8_t *,              8);

    assign(s->above_intra_ctx,     uint8_t *,              8);

    assign(s->above_comp_ctx,      uint8_t *,              8);

    assign(s->above_ref_ctx,       uint8_t *,              8);

    assign(s->above_filter_ctx,    uint8_t *,              8);

    assign(s->lflvl,               struct VP9Filter *,     1);

    assign(s->above_mv_ctx,        VP56mv(*)[2],          16);

    assign(s->segmentation_map,    uint8_t *,             64 * s->sb_rows);

    assign(s->mv[0],               struct VP9mvrefPair *, 64 * s->sb_rows);

    assign(s->mv[1],               struct VP9mvrefPair *, 64 * s->sb_rows);

#undef assign



    return 0;

}
