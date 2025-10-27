static int mov_read_trak(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    MOVStreamContext *sc;

    int ret;



    st = avformat_new_stream(c->fc, NULL);

    if (!st) return AVERROR(ENOMEM);

    st->id = c->fc->nb_streams;

    sc = av_mallocz(sizeof(MOVStreamContext));

    if (!sc) return AVERROR(ENOMEM);



    st->priv_data = sc;

    st->codec->codec_type = AVMEDIA_TYPE_DATA;

    sc->ffindex = st->index;



    if ((ret = mov_read_default(c, pb, atom)) < 0)

        return ret;



    /* sanity checks */

    if (sc->chunk_count && (!sc->stts_count || !sc->stsc_count ||

                            (!sc->sample_size && !sc->sample_count))) {

        av_log(c->fc, AV_LOG_ERROR, "stream %d, missing mandatory atoms, broken header\n",

               st->index);

        return 0;

    }



    fix_timescale(c, sc);



    avpriv_set_pts_info(st, 64, 1, sc->time_scale);



    mov_build_index(c, st);



    if (sc->dref_id-1 < sc->drefs_count && sc->drefs[sc->dref_id-1].path) {

        MOVDref *dref = &sc->drefs[sc->dref_id - 1];

        if (mov_open_dref(&sc->pb, c->fc->filename, dref, &c->fc->interrupt_callback,

            c->use_absolute_path, c->fc) < 0)

            av_log(c->fc, AV_LOG_ERROR,

                   "stream %d, error opening alias: path='%s', dir='%s', "

                   "filename='%s', volume='%s', nlvl_from=%d, nlvl_to=%d\n",

                   st->index, dref->path, dref->dir, dref->filename,

                   dref->volume, dref->nlvl_from, dref->nlvl_to);

    } else {

        sc->pb = c->fc->pb;

        sc->pb_is_copied = 1;

    }



    if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

        if (!st->sample_aspect_ratio.num &&

            (st->codec->width != sc->width || st->codec->height != sc->height)) {

            st->sample_aspect_ratio = av_d2q(((double)st->codec->height * sc->width) /

                                             ((double)st->codec->width * sc->height), INT_MAX);

        }



#if FF_API_R_FRAME_RATE

        if (sc->stts_count == 1 || (sc->stts_count == 2 && sc->stts_data[1].count == 1))

            av_reduce(&st->r_frame_rate.num, &st->r_frame_rate.den,

                      sc->time_scale, sc->stts_data[0].duration, INT_MAX);

#endif

    }



    // done for ai5q, ai52, ai55, ai1q, ai12 and ai15.

    if (!st->codec->extradata_size && st->codec->codec_id == AV_CODEC_ID_H264 &&

        TAG_IS_AVCI(st->codec->codec_tag)) {

        ret = ff_generate_avci_extradata(st);

        if (ret < 0)

            return ret;

    }



    switch (st->codec->codec_id) {

#if CONFIG_H261_DECODER

    case AV_CODEC_ID_H261:

#endif

#if CONFIG_H263_DECODER

    case AV_CODEC_ID_H263:

#endif

#if CONFIG_MPEG4_DECODER

    case AV_CODEC_ID_MPEG4:

#endif

        st->codec->width = 0; /* let decoder init width/height */

        st->codec->height= 0;

        break;

    }



    /* Do not need those anymore. */

    av_freep(&sc->chunk_offsets);

    av_freep(&sc->stsc_data);

    av_freep(&sc->sample_sizes);

    av_freep(&sc->keyframes);

    av_freep(&sc->stts_data);

    av_freep(&sc->stps_data);

    av_freep(&sc->elst_data);

    av_freep(&sc->rap_group);



    return 0;

}
