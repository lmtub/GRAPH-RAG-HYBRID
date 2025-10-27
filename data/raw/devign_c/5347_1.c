static int send_sub_rect_nojpeg(VncState *vs, int x, int y, int w, int h,

                                int bg, int fg, int colors, VncPalette *palette)

{

    int ret;



    if (colors == 0) {

        if (tight_detect_smooth_image(vs, w, h)) {

            ret = send_gradient_rect(vs, x, y, w, h);


            ret = send_full_color_rect(vs, x, y, w, h);

        }

    } else if (colors == 1) {

        ret = send_solid_rect(vs);

    } else if (colors == 2) {

        ret = send_mono_rect(vs, x, y, w, h, bg, fg);

    } else if (colors <= 256) {

        ret = send_palette_rect(vs, x, y, w, h, palette);



    }

    return ret;

}