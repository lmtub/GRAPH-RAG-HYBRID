ogg_get_length (AVFormatContext * s)

{

    ogg_t *ogg = s->priv_data;

    int idx = -1, i;

    offset_t size, end;



    if(s->pb.is_streamed)

        return 0;



// already set

    if (s->duration != AV_NOPTS_VALUE)

        return 0;



    size = url_fsize(&s->pb);

    if(size < 0)

        return 0;

    end = size > MAX_PAGE_SIZE? size - MAX_PAGE_SIZE: size;



    ogg_save (s);

    url_fseek (&s->pb, end, SEEK_SET);



    while (!ogg_read_page (s, &i)){

        if (ogg->streams[i].granule != -1 && ogg->streams[i].granule != 0)

            idx = i;

    }



    if (idx != -1){

        s->streams[idx]->duration =

            ogg_gptopts (s, idx, ogg->streams[idx].granule);

    }



    ogg->size = size;

    ogg_restore (s, 0);

    ogg_save (s);

    while (!ogg_read_page (s, &i)) {

        if (i == idx && ogg->streams[i].granule != -1 && ogg->streams[i].granule != 0)

            break;

    }

    if (i == idx) {

        s->streams[idx]->start_time = ogg_gptopts (s, idx, ogg->streams[idx].granule);

        s->streams[idx]->duration -= s->streams[idx]->start_time;

    }

    ogg_restore (s, 0);



    return 0;

}
