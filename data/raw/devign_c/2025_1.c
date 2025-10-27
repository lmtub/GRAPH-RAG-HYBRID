static void ogg_write_pages(AVFormatContext *s, int flush)

{

    OGGContext *ogg = s->priv_data;

    OGGPageList *next, *p;



    if (!ogg->page_list)

        return;



    for (p = ogg->page_list; p; ) {

        OGGStreamContext *oggstream =

            s->streams[p->page.stream_index]->priv_data;

        if (oggstream->page_count < 2 && !flush)

            break;

        ogg_write_page(s, &p->page,

                       flush && oggstream->page_count == 1 ? 4 : 0); // eos

        next = p->next;

        av_freep(&p);

        p = next;

    }

    ogg->page_list = p;

}
