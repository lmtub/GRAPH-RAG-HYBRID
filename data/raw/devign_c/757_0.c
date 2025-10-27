static inline void mct_decode(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile)

{

    int i, csize = 1;

    void *src[3];



    for (i = 1; i < 3; i++)

        if (tile->codsty[0].transform != tile->codsty[i].transform) {

            av_log(s->avctx, AV_LOG_ERROR, "Transforms mismatch, MCT not supported\n");

            return;

        }



    for (i = 0; i < 3; i++)

        if (tile->codsty[0].transform == FF_DWT97)

            src[i] = tile->comp[i].f_data;

        else

            src[i] = tile->comp[i].i_data;



    for (i = 0; i < 2; i++)

        csize *= tile->comp[0].coord[i][1] - tile->comp[0].coord[i][0];



    s->dsp.mct_decode[tile->codsty[0].transform](src[0], src[1], src[2], csize);

}
