av_cold void ff_diracdsp_init(DiracDSPContext *c)

{

    c->dirac_hpel_filter = dirac_hpel_filter;

    c->add_rect_clamped = add_rect_clamped_c;

    c->put_signed_rect_clamped[0] = put_signed_rect_clamped_8bit_c;

    c->put_signed_rect_clamped[1] = put_signed_rect_clamped_10bit_c;




    c->add_dirac_obmc[0] = add_obmc8_c;

    c->add_dirac_obmc[1] = add_obmc16_c;

    c->add_dirac_obmc[2] = add_obmc32_c;



    c->weight_dirac_pixels_tab[0] = weight_dirac_pixels8_c;

    c->weight_dirac_pixels_tab[1] = weight_dirac_pixels16_c;

    c->weight_dirac_pixels_tab[2] = weight_dirac_pixels32_c;

    c->biweight_dirac_pixels_tab[0] = biweight_dirac_pixels8_c;

    c->biweight_dirac_pixels_tab[1] = biweight_dirac_pixels16_c;

    c->biweight_dirac_pixels_tab[2] = biweight_dirac_pixels32_c;



    PIXFUNC(put, 8);

    PIXFUNC(put, 16);

    PIXFUNC(put, 32);

    PIXFUNC(avg, 8);

    PIXFUNC(avg, 16);

    PIXFUNC(avg, 32);



    if (HAVE_MMX && HAVE_YASM) ff_diracdsp_init_mmx(c);

}