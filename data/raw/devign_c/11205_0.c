av_cold void ff_intrax8_common_init(IntraX8Context *w, MpegEncContext *const s)

{

    w->s = s;

    x8_vlc_init();

    assert(s->mb_width > 0);



    // two rows, 2 blocks per cannon mb

    w->prediction_table = av_mallocz(s->mb_width * 2 * 2);



    ff_init_scantable(s->idsp.idct_permutation, &w->scantable[0],

                      ff_wmv1_scantable[0]);

    ff_init_scantable(s->idsp.idct_permutation, &w->scantable[1],

                      ff_wmv1_scantable[2]);

    ff_init_scantable(s->idsp.idct_permutation, &w->scantable[2],

                      ff_wmv1_scantable[3]);



    ff_intrax8dsp_init(&w->dsp);

}
