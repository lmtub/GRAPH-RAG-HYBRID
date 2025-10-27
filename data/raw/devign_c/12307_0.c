static void find_best_solid_area(VncState *vs, int x, int y, int w, int h,

                                 uint32_t color, int *w_ptr, int *h_ptr)

{

    int dx, dy, dw, dh;

    int w_prev;

    int w_best = 0, h_best = 0;



    w_prev = w;



    for (dy = y; dy < y + h; dy += VNC_TIGHT_MAX_SPLIT_TILE_SIZE) {



        dh = MIN(VNC_TIGHT_MAX_SPLIT_TILE_SIZE, y + h - dy);

        dw = MIN(VNC_TIGHT_MAX_SPLIT_TILE_SIZE, w_prev);



        if (!check_solid_tile(vs, x, dy, dw, dh, &color, true)) {

            break;

        }



        for (dx = x + dw; dx < x + w_prev;) {

            dw = MIN(VNC_TIGHT_MAX_SPLIT_TILE_SIZE, x + w_prev - dx);



            if (!check_solid_tile(vs, dx, dy, dw, dh, &color, true)) {

                break;

            }

            dx += dw;

        }



        w_prev = dx - x;

        if (w_prev * (dy + dh - y) > w_best * h_best) {

            w_best = w_prev;

            h_best = dy + dh - y;

        }

    }



    *w_ptr = w_best;

    *h_ptr = h_best;

}
