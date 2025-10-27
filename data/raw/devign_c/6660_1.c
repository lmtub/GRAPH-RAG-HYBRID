static void copy_cell(Indeo3DecodeContext *ctx, Plane *plane, Cell *cell)

{

    int     h, w, mv_x, mv_y, offset, offset_dst;

    uint8_t *src, *dst;



    /* setup output and reference pointers */

    offset_dst  = (cell->ypos << 2) * plane->pitch + (cell->xpos << 2);

    dst         = plane->pixels[ctx->buf_sel] + offset_dst;


    mv_y        = cell->mv_ptr[0];

    mv_x        = cell->mv_ptr[1];



    offset      = offset_dst + mv_y * plane->pitch + mv_x;

    src         = plane->pixels[ctx->buf_sel ^ 1] + offset;



    h = cell->height << 2;



    for (w = cell->width; w > 0;) {

        /* copy using 16xH blocks */

        if (!((cell->xpos << 2) & 15) && w >= 4) {

            for (; w >= 4; src += 16, dst += 16, w -= 4)

                ctx->dsp.put_no_rnd_pixels_tab[0][0](dst, src, plane->pitch, h);

        }



        /* copy using 8xH blocks */

        if (!((cell->xpos << 2) & 7) && w >= 2) {

            ctx->dsp.put_no_rnd_pixels_tab[1][0](dst, src, plane->pitch, h);

            w -= 2;

            src += 8;

            dst += 8;

        }



        if (w >= 1) {

            copy_block4(dst, src, plane->pitch, plane->pitch, h);

            w--;

            src += 4;

            dst += 4;

        }

    }

}