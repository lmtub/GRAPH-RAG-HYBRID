static int mpegps_read_pes_header(AVFormatContext *s,

                                  int64_t *ppos, int *pstart_code,

                                  int64_t *ppts, int64_t *pdts)

{

    MpegDemuxContext *m = s->priv_data;

    int len, size, startcode, c, flags, header_len;

    int64_t pts, dts, last_pos;



    last_pos = -1;

 redo:

        /* next start code (should be immediately after) */

        m->header_state = 0xff;

        size = MAX_SYNC_SIZE;

        startcode = find_next_start_code(&s->pb, &size, &m->header_state);

    //printf("startcode=%x pos=0x%"PRIx64"\n", startcode, url_ftell(&s->pb));

    if (startcode < 0)

        return AVERROR_IO;

    if (startcode == PACK_START_CODE)

        goto redo;

    if (startcode == SYSTEM_HEADER_START_CODE)

        goto redo;

    if (startcode == PADDING_STREAM ||

        startcode == PRIVATE_STREAM_2) {

        /* skip them */

        len = get_be16(&s->pb);

        url_fskip(&s->pb, len);

        goto redo;

    }

    if (startcode == PROGRAM_STREAM_MAP) {

        mpegps_psm_parse(m, &s->pb);

        goto redo;

    }



    /* find matching stream */

    if (!((startcode >= 0x1c0 && startcode <= 0x1df) ||

          (startcode >= 0x1e0 && startcode <= 0x1ef) ||

          (startcode == 0x1bd)))

        goto redo;

    if (ppos) {

        *ppos = url_ftell(&s->pb) - 4;

    }

    len = get_be16(&s->pb);

    pts = AV_NOPTS_VALUE;

    dts = AV_NOPTS_VALUE;

    /* stuffing */

    for(;;) {

        if (len < 1)

            goto redo;

        c = get_byte(&s->pb);

        len--;

        /* XXX: for mpeg1, should test only bit 7 */

        if (c != 0xff)

            break;

    }

    if ((c & 0xc0) == 0x40) {

        /* buffer scale & size */

        if (len < 2)

            goto redo;

        get_byte(&s->pb);

        c = get_byte(&s->pb);

        len -= 2;

    }

    if ((c & 0xf0) == 0x20) {

        if (len < 4)

            goto redo;

        dts = pts = get_pts(&s->pb, c);

        len -= 4;

    } else if ((c & 0xf0) == 0x30) {

        if (len < 9)

            goto redo;

        pts = get_pts(&s->pb, c);

        dts = get_pts(&s->pb, -1);

        len -= 9;

    } else if ((c & 0xc0) == 0x80) {

        /* mpeg 2 PES */

#if 0 /* some streams have this field set for no apparent reason */

        if ((c & 0x30) != 0) {

            /* Encrypted multiplex not handled */

            goto redo;

        }

#endif

        flags = get_byte(&s->pb);

        header_len = get_byte(&s->pb);

        len -= 2;

        if (header_len > len)

            goto redo;

        if ((flags & 0xc0) == 0x80) {

            dts = pts = get_pts(&s->pb, -1);

            if (header_len < 5)

                goto redo;

            header_len -= 5;

            len -= 5;

        } if ((flags & 0xc0) == 0xc0) {

            pts = get_pts(&s->pb, -1);

            dts = get_pts(&s->pb, -1);

            if (header_len < 10)

                goto redo;

            header_len -= 10;

            len -= 10;

        }

        len -= header_len;

        while (header_len > 0) {

            get_byte(&s->pb);

            header_len--;

        }

    }

    else if( c!= 0xf )

        goto redo;



    if (startcode == PRIVATE_STREAM_1 && !m->psm_es_type[startcode & 0xff]) {

        if (len < 1)

            goto redo;

        startcode = get_byte(&s->pb);

        len--;

        if (startcode >= 0x80 && startcode <= 0xbf) {

            /* audio: skip header */

            if (len < 3)

                goto redo;

            get_byte(&s->pb);

            get_byte(&s->pb);

            get_byte(&s->pb);

            len -= 3;

        }

    }

    if(dts != AV_NOPTS_VALUE && ppos){

        int i;

        for(i=0; i<s->nb_streams; i++){

            if(startcode == s->streams[i]->id) {

                av_add_index_entry(s->streams[i], *ppos, dts, 0, 0, AVINDEX_KEYFRAME /* FIXME keyframe? */);

            }

        }

    }



    *pstart_code = startcode;

    *ppts = pts;

    *pdts = dts;

    return len;

}
