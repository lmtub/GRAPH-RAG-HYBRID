void ff_rv34dsp_init_neon(RV34DSPContext *c, DSPContext* dsp)

{

    c->rv34_inv_transform    = ff_rv34_inv_transform_noround_neon;

    c->rv34_inv_transform_dc = ff_rv34_inv_transform_noround_dc_neon;



    c->rv34_idct_add    = ff_rv34_idct_add_neon;

    c->rv34_idct_dc_add = ff_rv34_idct_dc_add_neon;

}
