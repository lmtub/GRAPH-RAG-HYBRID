av_cold void ff_wmv2_common_init(Wmv2Context * w){

    MpegEncContext * const s= &w->s;



    ff_init_scantable(s->dsp.idct_permutation, &w->abt_scantable[0], ff_wmv2_scantableA);

    ff_init_scantable(s->dsp.idct_permutation, &w->abt_scantable[1], ff_wmv2_scantableB);

}
