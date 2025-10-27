void dsputil_init_mmi(void)

{

    clear_blocks = clear_blocks_mmi;

    

    put_pixels_tab[1][0] = put_pixels8_mmi;

    put_no_rnd_pixels_tab[1][0] = put_pixels8_mmi;

    

    put_pixels_tab[0][0] = put_pixels16_mmi;

    put_no_rnd_pixels_tab[0][0] = put_pixels16_mmi;

    

    get_pixels = get_pixels_mmi;

}
