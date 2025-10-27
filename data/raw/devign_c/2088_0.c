static int parse_bintree(Indeo3DecodeContext *ctx, AVCodecContext *avctx,

                         Plane *plane, int code, Cell *ref_cell,

                         const int depth, const int strip_width)

{

    Cell    curr_cell;

    int     bytes_used;



    if (depth <= 0) {

        av_log(avctx, AV_LOG_ERROR, "Stack overflow (corrupted binary tree)!\n");

        return AVERROR_INVALIDDATA; // unwind recursion

    }



    curr_cell = *ref_cell; // clone parent cell

    if (code == H_SPLIT) {

        SPLIT_CELL(ref_cell->height, curr_cell.height);

        ref_cell->ypos   += curr_cell.height;

        ref_cell->height -= curr_cell.height;

    } else if (code == V_SPLIT) {

        if (curr_cell.width > strip_width) {

            /* split strip */

            curr_cell.width = (curr_cell.width <= (strip_width << 1) ? 1 : 2) * strip_width;

        } else

            SPLIT_CELL(ref_cell->width, curr_cell.width);

        ref_cell->xpos  += curr_cell.width;

        ref_cell->width -= curr_cell.width;

    }



    while (1) { /* loop until return */

        RESYNC_BITSTREAM;

        switch (code = get_bits(&ctx->gb, 2)) {

        case H_SPLIT:

        case V_SPLIT:

            if (parse_bintree(ctx, avctx, plane, code, &curr_cell, depth - 1, strip_width))

                return AVERROR_INVALIDDATA;

            break;

        case INTRA_NULL:

            if (!curr_cell.tree) { /* MC tree INTRA code */

                curr_cell.mv_ptr = 0; /* mark the current strip as INTRA */

                curr_cell.tree   = 1; /* enter the VQ tree */

            } else { /* VQ tree NULL code */

                RESYNC_BITSTREAM;

                code = get_bits(&ctx->gb, 2);

                if (code >= 2) {

                    av_log(avctx, AV_LOG_ERROR, "Invalid VQ_NULL code: %d\n", code);

                    return AVERROR_INVALIDDATA;

                }

                if (code == 1)

                    av_log(avctx, AV_LOG_ERROR, "SkipCell procedure not implemented yet!\n");



                CHECK_CELL

                copy_cell(ctx, plane, &curr_cell);

                return 0;

            }

            break;

        case INTER_DATA:

            if (!curr_cell.tree) { /* MC tree INTER code */

                /* get motion vector index and setup the pointer to the mv set */

                if (!ctx->need_resync)

                    ctx->next_cell_data = &ctx->gb.buffer[(get_bits_count(&ctx->gb) + 7) >> 3];

                curr_cell.mv_ptr = &ctx->mc_vectors[*(ctx->next_cell_data++) << 1];

                curr_cell.tree   = 1; /* enter the VQ tree */

                UPDATE_BITPOS(8);

            } else { /* VQ tree DATA code */

                if (!ctx->need_resync)

                    ctx->next_cell_data = &ctx->gb.buffer[(get_bits_count(&ctx->gb) + 7) >> 3];



                CHECK_CELL

                bytes_used = decode_cell(ctx, avctx, plane, &curr_cell,

                                         ctx->next_cell_data, ctx->last_byte);

                if (bytes_used < 0)

                    return AVERROR_INVALIDDATA;



                UPDATE_BITPOS(bytes_used << 3);

                ctx->next_cell_data += bytes_used;

                return 0;

            }

            break;

        }

    }//while



    return 0;

}
