static int mov_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    MOVContext *mov = s->priv_data;

    MOVStreamContext *sc;

    AVIndexEntry *sample;

    AVStream *st = NULL;

    int ret;

    mov->fc = s;

 retry:

    sample = mov_find_next_sample(s, &st);

    if (!sample) {

        mov->found_mdat = 0;

        if (!mov->next_root_atom)

            return AVERROR_EOF;

        avio_seek(s->pb, mov->next_root_atom, SEEK_SET);

        mov->next_root_atom = 0;

        if (mov_read_default(mov, s->pb, (MOVAtom){ AV_RL32("root"), INT64_MAX }) < 0 ||

            avio_feof(s->pb))

            return AVERROR_EOF;

        av_log(s, AV_LOG_TRACE, "read fragments, offset 0x%"PRIx64"\n", avio_tell(s->pb));

        goto retry;

    }

    sc = st->priv_data;

    /* must be done just before reading, to avoid infinite loop on sample */

    sc->current_sample++;



    if (mov->next_root_atom) {

        sample->pos = FFMIN(sample->pos, mov->next_root_atom);

        sample->size = FFMIN(sample->size, (mov->next_root_atom - sample->pos));

    }



    if (st->discard != AVDISCARD_ALL) {

        if (avio_seek(sc->pb, sample->pos, SEEK_SET) != sample->pos) {

            av_log(mov->fc, AV_LOG_ERROR, "stream %d, offset 0x%"PRIx64": partial file\n",

                   sc->ffindex, sample->pos);

            return AVERROR_INVALIDDATA;

        }

        ret = av_get_packet(sc->pb, pkt, sample->size);

        if (ret < 0)

            return ret;

        if (sc->has_palette) {

            uint8_t *pal;



            pal = av_packet_new_side_data(pkt, AV_PKT_DATA_PALETTE, AVPALETTE_SIZE);

            if (!pal) {

                av_log(mov->fc, AV_LOG_ERROR, "Cannot append palette to packet\n");

            } else {

                memcpy(pal, sc->palette, AVPALETTE_SIZE);

                sc->has_palette = 0;

            }

        }

#if CONFIG_DV_DEMUXER

        if (mov->dv_demux && sc->dv_audio_container) {

            avpriv_dv_produce_packet(mov->dv_demux, pkt, pkt->data, pkt->size, pkt->pos);

            av_freep(&pkt->data);

            pkt->size = 0;

            ret = avpriv_dv_get_packet(mov->dv_demux, pkt);

            if (ret < 0)

                return ret;

        }

#endif

    }



    pkt->stream_index = sc->ffindex;

    pkt->dts = sample->timestamp;

    if (sc->ctts_data && sc->ctts_index < sc->ctts_count) {

        pkt->pts = pkt->dts + sc->dts_shift + sc->ctts_data[sc->ctts_index].duration;

        /* update ctts context */

        sc->ctts_sample++;

        if (sc->ctts_index < sc->ctts_count &&

            sc->ctts_data[sc->ctts_index].count == sc->ctts_sample) {

            sc->ctts_index++;

            sc->ctts_sample = 0;

        }

        if (sc->wrong_dts)

            pkt->dts = AV_NOPTS_VALUE;

    } else {

        int64_t next_dts = (sc->current_sample < st->nb_index_entries) ?

            st->index_entries[sc->current_sample].timestamp : st->duration;

        pkt->duration = next_dts - pkt->dts;

        pkt->pts = pkt->dts;

    }

    if (st->discard == AVDISCARD_ALL)

        goto retry;

    pkt->flags |= sample->flags & AVINDEX_KEYFRAME ? AV_PKT_FLAG_KEY : 0;

    pkt->pos = sample->pos;



    return 0;

}
