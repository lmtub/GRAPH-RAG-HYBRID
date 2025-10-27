static av_cold void alloc_temp(HYuvContext *s)

{

    int i;



    if (s->bitstream_bpp<24) {

        for (i=0; i<3; i++) {

            s->temp[i]= av_malloc(s->width + 16);

        }

    } else {

        s->temp[0]= av_mallocz(4*s->width + 16);

    }

}
