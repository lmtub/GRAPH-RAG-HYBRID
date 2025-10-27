static av_cold void build_modpred(Indeo3DecodeContext *s)

{

  int i, j;



  s->ModPred = av_malloc(8 * 128);



  for (i=0; i < 128; ++i) {

    s->ModPred[i+0*128] = i >  126 ? 254 : 2*(i + 1 - ((i + 1) % 2));

    s->ModPred[i+1*128] = i ==   7 ?  20 :

                          i == 119 ||

                          i == 120 ? 236 : 2*(i + 2 - ((i + 1) % 3));

    s->ModPred[i+2*128] = i >  125 ? 248 : 2*(i + 2 - ((i + 2) % 4));

    s->ModPred[i+3*128] =                  2*(i + 1 - ((i - 3) % 5));

    s->ModPred[i+4*128] = i ==   8 ?  20 : 2*(i + 1 - ((i - 3) % 6));

    s->ModPred[i+5*128] =                  2*(i + 4 - ((i + 3) % 7));

    s->ModPred[i+6*128] = i >  123 ? 240 : 2*(i + 4 - ((i + 4) % 8));

    s->ModPred[i+7*128] =                  2*(i + 5 - ((i + 4) % 9));

  }



  s->corrector_type = av_malloc(24 * 256);



  for (i=0; i < 24; ++i) {

    for (j=0; j < 256; ++j) {

      s->corrector_type[i*256+j] = j < corrector_type_0[i]          ? 1 :

                                   j < 248 || (i == 16 && j == 248) ? 0 :

                                   corrector_type_2[j - 248];

    }

  }

}
