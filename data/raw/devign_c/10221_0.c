static int find_slice_quant(AVCodecContext *avctx, const AVFrame *pic,

                            int trellis_node, int x, int y, int mbs_per_slice)

{

    ProresContext *ctx = avctx->priv_data;

    int i, q, pq, xp, yp;

    const uint16_t *src;

    int slice_width_factor = av_log2(mbs_per_slice);

    int num_cblocks[MAX_PLANES], pwidth;

    int plane_factor[MAX_PLANES], is_chroma[MAX_PLANES];

    const int min_quant = ctx->profile_info->min_quant;

    const int max_quant = ctx->profile_info->max_quant;

    int error, bits, bits_limit;

    int mbs, prev, cur, new_score;

    int slice_bits[TRELLIS_WIDTH], slice_score[TRELLIS_WIDTH];



    mbs = x + mbs_per_slice;



    for (i = 0; i < ctx->num_planes; i++) {

        is_chroma[i]    = (i == 1 || i == 2);

        plane_factor[i] = slice_width_factor + 2;

        if (is_chroma[i])

            plane_factor[i] += ctx->chroma_factor - 3;

        if (!is_chroma[i] || ctx->chroma_factor == CFACTOR_Y444) {

            xp             = x << 4;

            yp             = y << 4;

            num_cblocks[i] = 4;

            pwidth         = avctx->width;

        } else {

            xp             = x << 3;

            yp             = y << 4;

            num_cblocks[i] = 2;

            pwidth         = avctx->width >> 1;

        }

        src = (const uint16_t*)(pic->data[i] + yp * pic->linesize[i]) + xp;



        get_slice_data(ctx, src, pic->linesize[i], xp, yp,

                       pwidth, avctx->height, ctx->blocks[i],

                       mbs_per_slice, num_cblocks[i]);

    }



    for (q = min_quant; q <= max_quant; q++) {

        ctx->nodes[trellis_node + q].prev_node = -1;

        ctx->nodes[trellis_node + q].quant     = q;

    }



    // todo: maybe perform coarser quantising to fit into frame size when needed

    for (q = min_quant; q <= max_quant; q++) {

        bits  = 0;

        error = 0;

        for (i = 0; i < ctx->num_planes; i++) {

            bits += estimate_slice_plane(ctx, &error, i,

                                         src, pic->linesize[i],

                                         mbs_per_slice,

                                         num_cblocks[i], plane_factor[i],

                                         ctx->quants[q]);

        }

        if (bits > 65000 * 8) {

            error = SCORE_LIMIT;

            break;

        }

        slice_bits[q]  = bits;

        slice_score[q] = error;

    }



    bits_limit = mbs * ctx->bits_per_mb;

    for (pq = min_quant; pq <= max_quant; pq++) {

        prev = trellis_node - TRELLIS_WIDTH + pq;



        for (q = min_quant; q <= max_quant; q++) {

            cur = trellis_node + q;



            bits  = ctx->nodes[prev].bits + slice_bits[q];

            error = slice_score[q];

            if (bits > bits_limit)

                error = SCORE_LIMIT;



            if (ctx->nodes[prev].score < SCORE_LIMIT && error < SCORE_LIMIT)

                new_score = ctx->nodes[prev].score + error;

            else

                new_score = SCORE_LIMIT;

            if (ctx->nodes[cur].prev_node == -1 ||

                ctx->nodes[cur].score >= new_score) {



                ctx->nodes[cur].bits      = bits;

                ctx->nodes[cur].score     = new_score;

                ctx->nodes[cur].prev_node = prev;

            }

        }

    }



    error = ctx->nodes[trellis_node + min_quant].score;

    pq    = trellis_node + min_quant;

    for (q = min_quant + 1; q <= max_quant; q++) {

        if (ctx->nodes[trellis_node + q].score <= error) {

            error = ctx->nodes[trellis_node + q].score;

            pq    = trellis_node + q;

        }

    }



    return pq;

}
