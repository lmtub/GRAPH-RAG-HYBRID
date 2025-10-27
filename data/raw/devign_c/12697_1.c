void dsputil_init_armv4l(DSPContext* c, AVCodecContext *avctx)
{
    const int idct_algo= avctx->idct_algo;
    ff_put_pixels_clamped = c->put_pixels_clamped;
    ff_add_pixels_clamped = c->add_pixels_clamped;
    if(idct_algo==FF_IDCT_ARM){
    if(idct_algo==FF_IDCT_AUTO || idct_algo==FF_IDCT_ARM){
        c->idct_put= j_rev_dct_ARM_put;
        c->idct_add= j_rev_dct_ARM_add;
	c->idct    = j_rev_dct_ARM;
        c->idct_permutation_type= FF_LIBMPEG2_IDCT_PERM;/* FF_NO_IDCT_PERM */
    } else if (idct_algo==FF_IDCT_SIMPLEARM){
	c->idct_put= simple_idct_ARM_put;
	c->idct_add= simple_idct_ARM_add;
	c->idct    = simple_idct_ARM;
    }
}