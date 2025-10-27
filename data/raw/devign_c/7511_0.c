static int decode_cell(Indeo3DecodeContext *ctx, AVCodecContext *avctx,

                       Plane *plane, Cell *cell, const uint8_t *data_ptr,

                       const uint8_t *last_ptr)

{

    int           x, mv_x, mv_y, mode, vq_index, prim_indx, second_indx;

    int           zoom_fac;

    int           offset, error = 0, swap_quads[2];

    uint8_t       code, *block, *ref_block = 0;

    const vqEntry *delta[2];

    const uint8_t *data_start = data_ptr;



    /* get coding mode and VQ table index from the VQ descriptor byte */

    code     = *data_ptr++;

    mode     = code >> 4;

    vq_index = code & 0xF;



    /* setup output and reference pointers */

    offset = (cell->ypos << 2) * plane->pitch + (cell->xpos << 2);

    block  =  plane->pixels[ctx->buf_sel] + offset;

    if (!cell->mv_ptr) {

        /* use previous line as reference for INTRA cells */

        ref_block = block - plane->pitch;

    } else if (mode >= 10) {

        /* for mode 10 and 11 INTER first copy the predicted cell into the current one */

        /* so we don't need to do data copying for each RLE code later */

        copy_cell(ctx, plane, cell);

    } else {

        /* set the pointer to the reference pixels for modes 0-4 INTER */

        mv_y      = cell->mv_ptr[0];

        mv_x      = cell->mv_ptr[1];

        if (   mv_x + 4*cell->xpos < 0

            || mv_y + 4*cell->ypos < 0

            || mv_x + 4*cell->xpos + 4*cell->width  > plane->width

            || mv_y + 4*cell->ypos + 4*cell->height > plane->height) {

            av_log(avctx, AV_LOG_ERROR, "motion vector %d %d outside reference\n", mv_x + 4*cell->xpos, mv_y + 4*cell->ypos);

            return AVERROR_INVALIDDATA;

        }

        offset   += mv_y * plane->pitch + mv_x;

        ref_block = plane->pixels[ctx->buf_sel ^ 1] + offset;

    }



    /* select VQ tables as follows: */

    /* modes 0 and 3 use only the primary table for all lines in a block */

    /* while modes 1 and 4 switch between primary and secondary tables on alternate lines */

    if (mode == 1 || mode == 4) {

        code        = ctx->alt_quant[vq_index];

        prim_indx   = (code >> 4)  + ctx->cb_offset;

        second_indx = (code & 0xF) + ctx->cb_offset;

    } else {

        vq_index += ctx->cb_offset;

        prim_indx = second_indx = vq_index;

    }



    if (prim_indx >= 24 || second_indx >= 24) {

        av_log(avctx, AV_LOG_ERROR, "Invalid VQ table indexes! Primary: %d, secondary: %d!\n",

               prim_indx, second_indx);

        return AVERROR_INVALIDDATA;

    }



    delta[0] = &vq_tab[second_indx];

    delta[1] = &vq_tab[prim_indx];

    swap_quads[0] = second_indx >= 16;

    swap_quads[1] = prim_indx   >= 16;



    /* requantize the prediction if VQ index of this cell differs from VQ index */

    /* of the predicted cell in order to avoid overflows. */

    if (vq_index >= 8 && ref_block) {

        for (x = 0; x < cell->width << 2; x++)

            ref_block[x] = requant_tab[vq_index & 7][ref_block[x] & 127];

    }



    error = IV3_NOERR;



    switch (mode) {

    case 0: /*------------------ MODES 0 & 1 (4x4 block processing) --------------------*/

    case 1:

    case 3: /*------------------ MODES 3 & 4 (4x8 block processing) --------------------*/

    case 4:

        if (mode >= 3 && cell->mv_ptr) {

            av_log(avctx, AV_LOG_ERROR, "Attempt to apply Mode 3/4 to an INTER cell!\n");

            return AVERROR_INVALIDDATA;

        }



        zoom_fac = mode >= 3;

        error = decode_cell_data(cell, block, ref_block, plane->pitch, 0, zoom_fac,

                                 mode, delta, swap_quads, &data_ptr, last_ptr);

        break;

    case 10: /*-------------------- MODE 10 (8x8 block processing) ---------------------*/

    case 11: /*----------------- MODE 11 (4x8 INTER block processing) ------------------*/

        if (mode == 10 && !cell->mv_ptr) { /* MODE 10 INTRA processing */

            error = decode_cell_data(cell, block, ref_block, plane->pitch, 1, 1,

                                     mode, delta, swap_quads, &data_ptr, last_ptr);

        } else { /* mode 10 and 11 INTER processing */

            if (mode == 11 && !cell->mv_ptr) {

               av_log(avctx, AV_LOG_ERROR, "Attempt to use Mode 11 for an INTRA cell!\n");

               return AVERROR_INVALIDDATA;

            }



            zoom_fac = mode == 10;

            error = decode_cell_data(cell, block, ref_block, plane->pitch,

                                     zoom_fac, 1, mode, delta, swap_quads,

                                     &data_ptr, last_ptr);

        }

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unsupported coding mode: %d\n", mode);

        return AVERROR_INVALIDDATA;

    }//switch mode



    switch (error) {

    case IV3_BAD_RLE:

        av_log(avctx, AV_LOG_ERROR, "Mode %d: RLE code %X is not allowed at the current line\n",

               mode, data_ptr[-1]);

        return AVERROR_INVALIDDATA;

    case IV3_BAD_DATA:

        av_log(avctx, AV_LOG_ERROR, "Mode %d: invalid VQ data\n", mode);

        return AVERROR_INVALIDDATA;

    case IV3_BAD_COUNTER:

        av_log(avctx, AV_LOG_ERROR, "Mode %d: RLE-FB invalid counter: %d\n", mode, code);

        return AVERROR_INVALIDDATA;

    case IV3_UNSUPPORTED:

        av_log(avctx, AV_LOG_ERROR, "Mode %d: unsupported RLE code: %X\n", mode, data_ptr[-1]);

        return AVERROR_INVALIDDATA;

    case IV3_OUT_OF_DATA:

        av_log(avctx, AV_LOG_ERROR, "Mode %d: attempt to read past end of buffer\n", mode);

        return AVERROR_INVALIDDATA;

    }



    return data_ptr - data_start; /* report number of bytes consumed from the input buffer */

}
