static int common_end(HYuvContext *s)

{

    int i;



    for(i = 0; i < 3; i++) {

        av_freep(&s->temp[i]);

    }

    return 0;

}
