void dsputil_init_mlib(void)

{

    put_pixels_tab[0][0] = put_pixels16_mlib;

    put_pixels_tab[0][1] = put_pixels16_x2_mlib;

    put_pixels_tab[0][2] = put_pixels16_y2_mlib;

    put_pixels_tab[0][3] = put_pixels16_xy2_mlib;

    put_pixels_tab[1][0] = put_pixels8_mlib;

    put_pixels_tab[1][1] = put_pixels8_x2_mlib;

    put_pixels_tab[1][2] = put_pixels8_y2_mlib;

    put_pixels_tab[1][3] = put_pixels8_xy2_mlib;



    avg_pixels_tab[0][0] = avg_pixels16_mlib;

    avg_pixels_tab[0][1] = avg_pixels16_x2_mlib;

    avg_pixels_tab[0][2] = avg_pixels16_y2_mlib;

    avg_pixels_tab[0][3] = avg_pixels16_xy2_mlib;

    avg_pixels_tab[1][0] = avg_pixels8_mlib;

    avg_pixels_tab[1][1] = avg_pixels8_x2_mlib;

    avg_pixels_tab[1][2] = avg_pixels8_y2_mlib;

    avg_pixels_tab[1][3] = avg_pixels8_xy2_mlib;

    

    put_no_rnd_pixels_tab[0][0] = put_pixels16_mlib;

    put_no_rnd_pixels_tab[1][0] = put_pixels8_mlib;

    

    add_pixels_clamped = add_pixels_clamped_mlib;

}
