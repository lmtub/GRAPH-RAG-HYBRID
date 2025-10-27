static void mov_text_cleanup_ftab(MovTextContext *m)

{

    int i;

    for(i = 0; i < m->count_f; i++) {

        av_freep(&m->ftab[i]->font);

        av_freep(&m->ftab[i]);

    }

    av_freep(&m->ftab);

}
