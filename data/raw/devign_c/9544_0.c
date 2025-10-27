static int dnxhd_decode_macroblock(DNXHDContext *ctx, AVFrame *frame,

                                   int x, int y)

{

    int shift1 = ctx->bit_depth == 10;

    int dct_linesize_luma   = frame->linesize[0];

    int dct_linesize_chroma = frame->linesize[1];

    uint8_t *dest_y, *dest_u, *dest_v;

    int dct_y_offset, dct_x_offset;

    int qscale, i;



    qscale = get_bits(&ctx->gb, 11);

    skip_bits1(&ctx->gb);



    if (qscale != ctx->last_qscale) {

        for (i = 0; i < 64; i++) {

            ctx->luma_scale[i]   = qscale * ctx->cid_table->luma_weight[i];

            ctx->chroma_scale[i] = qscale * ctx->cid_table->chroma_weight[i];

        }

        ctx->last_qscale = qscale;

    }



    for (i = 0; i < 8; i++) {

        ctx->bdsp.clear_block(ctx->blocks[i]);

        ctx->decode_dct_block(ctx, ctx->blocks[i], i, qscale);

    }

    if (ctx->is_444) {

        for (; i < 12; i++) {

            ctx->bdsp.clear_block(ctx->blocks[i]);

            ctx->decode_dct_block(ctx, ctx->blocks[i], i, qscale);

        }

    }



    if (frame->interlaced_frame) {

        dct_linesize_luma   <<= 1;

        dct_linesize_chroma <<= 1;

    }



    dest_y = frame->data[0] + ((y * dct_linesize_luma)   << 4) + (x << (4 + shift1));

    dest_u = frame->data[1] + ((y * dct_linesize_chroma) << 4) + (x << (3 + shift1 + ctx->is_444));

    dest_v = frame->data[2] + ((y * dct_linesize_chroma) << 4) + (x << (3 + shift1 + ctx->is_444));



    if (ctx->cur_field) {

        dest_y += frame->linesize[0];

        dest_u += frame->linesize[1];

        dest_v += frame->linesize[2];

    }



    dct_y_offset = dct_linesize_luma << 3;

    dct_x_offset = 8 << shift1;

    if (!ctx->is_444) {

        ctx->idsp.idct_put(dest_y,                               dct_linesize_luma, ctx->blocks[0]);

        ctx->idsp.idct_put(dest_y + dct_x_offset,                dct_linesize_luma, ctx->blocks[1]);

        ctx->idsp.idct_put(dest_y + dct_y_offset,                dct_linesize_luma, ctx->blocks[4]);

        ctx->idsp.idct_put(dest_y + dct_y_offset + dct_x_offset, dct_linesize_luma, ctx->blocks[5]);



        if (!(ctx->avctx->flags & CODEC_FLAG_GRAY)) {

            dct_y_offset = dct_linesize_chroma << 3;

            ctx->idsp.idct_put(dest_u,                dct_linesize_chroma, ctx->blocks[2]);

            ctx->idsp.idct_put(dest_v,                dct_linesize_chroma, ctx->blocks[3]);

            ctx->idsp.idct_put(dest_u + dct_y_offset, dct_linesize_chroma, ctx->blocks[6]);

            ctx->idsp.idct_put(dest_v + dct_y_offset, dct_linesize_chroma, ctx->blocks[7]);

        }

    } else {

        ctx->idsp.idct_put(dest_y,                               dct_linesize_luma, ctx->blocks[0]);

        ctx->idsp.idct_put(dest_y + dct_x_offset,                dct_linesize_luma, ctx->blocks[1]);

        ctx->idsp.idct_put(dest_y + dct_y_offset,                dct_linesize_luma, ctx->blocks[6]);

        ctx->idsp.idct_put(dest_y + dct_y_offset + dct_x_offset, dct_linesize_luma, ctx->blocks[7]);



        if (!(ctx->avctx->flags & CODEC_FLAG_GRAY)) {

            dct_y_offset = dct_linesize_chroma << 3;

            ctx->idsp.idct_put(dest_u,                               dct_linesize_chroma, ctx->blocks[2]);

            ctx->idsp.idct_put(dest_u + dct_x_offset,                dct_linesize_chroma, ctx->blocks[3]);

            ctx->idsp.idct_put(dest_u + dct_y_offset,                dct_linesize_chroma, ctx->blocks[8]);

            ctx->idsp.idct_put(dest_u + dct_y_offset + dct_x_offset, dct_linesize_chroma, ctx->blocks[9]);

            ctx->idsp.idct_put(dest_v,                               dct_linesize_chroma, ctx->blocks[4]);

            ctx->idsp.idct_put(dest_v + dct_x_offset,                dct_linesize_chroma, ctx->blocks[5]);

            ctx->idsp.idct_put(dest_v + dct_y_offset,                dct_linesize_chroma, ctx->blocks[10]);

            ctx->idsp.idct_put(dest_v + dct_y_offset + dct_x_offset, dct_linesize_chroma, ctx->blocks[11]);

        }

    }



    return 0;

}
