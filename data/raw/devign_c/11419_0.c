static int mov_read_trak(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    MOVStreamContext *sc;

    int ret;



    st = av_new_stream(c->fc, c->fc->nb_streams);

    if (!st) return AVERROR(ENOMEM);

    sc = av_mallocz(sizeof(MOVStreamContext));

    if (!sc) return AVERROR(ENOMEM);



    st->priv_data = sc;

    st->codec->codec_type = CODEC_TYPE_DATA;

    sc->ffindex = st->index;



    if ((ret = mov_read_default(c, pb, atom)) < 0)

        return ret;



    /* sanity checks */

    if(sc->chunk_count && (!sc->stts_count || !sc->stsc_count ||

                           (!sc->sample_size && !sc->sample_count))){

        av_log(c->fc, AV_LOG_ERROR, "stream %d, missing mandatory atoms, broken header\n",

               st->index);

        sc->sample_count = 0; //ignore track

        return 0;

    }

    if(!sc->time_rate)

        sc->time_rate=1;

    if(!sc->time_scale)

        sc->time_scale= c->time_scale;

    av_set_pts_info(st, 64, sc->time_rate, sc->time_scale);



    if (st->codec->codec_type == CODEC_TYPE_AUDIO &&

        !st->codec->frame_size && sc->stts_count == 1) {

        st->codec->frame_size = av_rescale(sc->stts_data[0].duration,

                                           st->codec->sample_rate, sc->time_scale);

        dprintf(c->fc, "frame size %d\n", st->codec->frame_size);

    }



    if(st->duration != AV_NOPTS_VALUE){

        assert(st->duration % sc->time_rate == 0);

        st->duration /= sc->time_rate;

    }



    mov_build_index(c, st);



    if (sc->dref_id-1 < sc->drefs_count && sc->drefs[sc->dref_id-1].path) {

        if (url_fopen(&sc->pb, sc->drefs[sc->dref_id-1].path, URL_RDONLY) < 0)

            av_log(c->fc, AV_LOG_ERROR, "stream %d, error opening file %s: %s\n",

                   st->index, sc->drefs[sc->dref_id-1].path, strerror(errno));

    } else

        sc->pb = c->fc->pb;



    switch (st->codec->codec_id) {

#if CONFIG_H261_DECODER

    case CODEC_ID_H261:

#endif

#if CONFIG_H263_DECODER

    case CODEC_ID_H263:

#endif

#if CONFIG_MPEG4_DECODER

    case CODEC_ID_MPEG4:

#endif

        st->codec->width= 0; /* let decoder init width/height */

        st->codec->height= 0;

        break;

    }



    /* Do not need those anymore. */

    av_freep(&sc->chunk_offsets);

    av_freep(&sc->stsc_data);

    av_freep(&sc->sample_sizes);

    av_freep(&sc->keyframes);

    av_freep(&sc->stts_data);



    return 0;

}
