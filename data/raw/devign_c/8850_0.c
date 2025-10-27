static int avi_write_header(AVFormatContext *s)

{

    AVIContext *avi = s->priv_data;

    AVIOContext *pb = s->pb;

    int bitrate, n, i, nb_frames, au_byterate, au_ssize, au_scale;

    AVCodecContext *stream, *video_enc;

    int64_t list1, list2, strh, strf;

    AVDictionaryEntry *t = NULL;



    if (s->nb_streams > AVI_MAX_STREAM_COUNT) {

        av_log(s, AV_LOG_ERROR, "AVI does not support >%d streams\n",

               AVI_MAX_STREAM_COUNT);

        return -1;

    }



    for (n = 0; n < s->nb_streams; n++) {

        s->streams[n]->priv_data = av_mallocz(sizeof(AVIStream));

        if (!s->streams[n]->priv_data)

            return AVERROR(ENOMEM);

    }



    /* header list */

    avi->riff_id = 0;

    list1 = avi_start_new_riff(s, pb, "AVI ", "hdrl");



    /* avi header */

    ffio_wfourcc(pb, "avih");

    avio_wl32(pb, 14 * 4);

    bitrate = 0;



    video_enc = NULL;

    for (n = 0; n < s->nb_streams; n++) {

        stream   = s->streams[n]->codec;

        bitrate += stream->bit_rate;

        if (stream->codec_type == AVMEDIA_TYPE_VIDEO)

            video_enc = stream;

    }



    nb_frames = 0;



    if (video_enc)

        avio_wl32(pb, (uint32_t) (INT64_C(1000000) * video_enc->time_base.num /

                                  video_enc->time_base.den));

    else

        avio_wl32(pb, 0);

    avio_wl32(pb, bitrate / 8); /* XXX: not quite exact */

    avio_wl32(pb, 0); /* padding */

    if (!pb->seekable)

        avio_wl32(pb, AVIF_TRUSTCKTYPE | AVIF_ISINTERLEAVED);  /* flags */

    else

        avio_wl32(pb, AVIF_TRUSTCKTYPE | AVIF_HASINDEX | AVIF_ISINTERLEAVED);  /* flags */

    avi->frames_hdr_all = avio_tell(pb); /* remember this offset to fill later */

    avio_wl32(pb, nb_frames); /* nb frames, filled later */

    avio_wl32(pb, 0); /* initial frame */

    avio_wl32(pb, s->nb_streams); /* nb streams */

    avio_wl32(pb, 1024 * 1024); /* suggested buffer size */

    if (video_enc) {

        avio_wl32(pb, video_enc->width);

        avio_wl32(pb, video_enc->height);

    } else {

        avio_wl32(pb, 0);

        avio_wl32(pb, 0);

    }

    avio_wl32(pb, 0); /* reserved */

    avio_wl32(pb, 0); /* reserved */

    avio_wl32(pb, 0); /* reserved */

    avio_wl32(pb, 0); /* reserved */



    /* stream list */

    for (i = 0; i < n; i++) {

        AVIStream *avist = s->streams[i]->priv_data;

        list2 = ff_start_tag(pb, "LIST");

        ffio_wfourcc(pb, "strl");



        stream = s->streams[i]->codec;



        /* stream generic header */

        strh = ff_start_tag(pb, "strh");

        switch (stream->codec_type) {

        case AVMEDIA_TYPE_SUBTITLE:

            // XSUB subtitles behave like video tracks, other subtitles

            // are not (yet) supported.

            if (stream->codec_id != AV_CODEC_ID_XSUB) {

                av_log(s, AV_LOG_ERROR,

                       "Subtitle streams other than DivX XSUB are not supported by the AVI muxer.\n");

                return AVERROR_PATCHWELCOME;

            }

        case AVMEDIA_TYPE_VIDEO:

            ffio_wfourcc(pb, "vids");

            break;

        case AVMEDIA_TYPE_AUDIO:

            ffio_wfourcc(pb, "auds");

            break;

//      case AVMEDIA_TYPE_TEXT:

//          ffio_wfourcc(pb, "txts");

//          break;

        case AVMEDIA_TYPE_DATA:

            ffio_wfourcc(pb, "dats");

            break;

        }

        if (stream->codec_type == AVMEDIA_TYPE_VIDEO ||

            stream->codec_id == AV_CODEC_ID_XSUB)

            avio_wl32(pb, stream->codec_tag);

        else

            avio_wl32(pb, 1);

        avio_wl32(pb, 0); /* flags */

        avio_wl16(pb, 0); /* priority */

        avio_wl16(pb, 0); /* language */

        avio_wl32(pb, 0); /* initial frame */



        ff_parse_specific_params(stream, &au_byterate, &au_ssize, &au_scale);



        avio_wl32(pb, au_scale); /* scale */

        avio_wl32(pb, au_byterate); /* rate */

        avpriv_set_pts_info(s->streams[i], 64, au_scale, au_byterate);



        avio_wl32(pb, 0); /* start */

        /* remember this offset to fill later */

        avist->frames_hdr_strm = avio_tell(pb);

        if (!pb->seekable)

            /* FIXME: this may be broken, but who cares */

            avio_wl32(pb, AVI_MAX_RIFF_SIZE);

        else

            avio_wl32(pb, 0);  /* length, XXX: filled later */



        /* suggested buffer size */ //FIXME set at the end to largest chunk

        if (stream->codec_type == AVMEDIA_TYPE_VIDEO)

            avio_wl32(pb, 1024 * 1024);

        else if (stream->codec_type == AVMEDIA_TYPE_AUDIO)

            avio_wl32(pb, 12 * 1024);

        else

            avio_wl32(pb, 0);

        avio_wl32(pb, -1); /* quality */

        avio_wl32(pb, au_ssize); /* sample size */

        avio_wl32(pb, 0);

        avio_wl16(pb, stream->width);

        avio_wl16(pb, stream->height);

        ff_end_tag(pb, strh);



        if (stream->codec_type != AVMEDIA_TYPE_DATA) {

            strf = ff_start_tag(pb, "strf");

            switch (stream->codec_type) {

            case AVMEDIA_TYPE_SUBTITLE:

                /* XSUB subtitles behave like video tracks, other subtitles

                 * are not (yet) supported. */

                if (stream->codec_id != AV_CODEC_ID_XSUB)

                    break;

            case AVMEDIA_TYPE_VIDEO:

                ff_put_bmp_header(pb, stream, ff_codec_bmp_tags, 0);

                break;

            case AVMEDIA_TYPE_AUDIO:

                if (ff_put_wav_header(pb, stream) < 0)

                    return -1;

                break;

            default:

                return -1;

            }

            ff_end_tag(pb, strf);

            if ((t = av_dict_get(s->streams[i]->metadata, "title", NULL, 0))) {

                ff_riff_write_info_tag(s->pb, "strn", t->value);

                t = NULL;

            }

        }



        if (pb->seekable) {

            unsigned char tag[5];

            int j;



            /* Starting to lay out AVI OpenDML master index.

             * We want to make it JUNK entry for now, since we'd

             * like to get away without making AVI an OpenDML one

             * for compatibility reasons. */

            avist->indexes.entry      = avist->indexes.ents_allocated = 0;

            avist->indexes.indx_start = ff_start_tag(pb, "JUNK");

            avio_wl16(pb, 4);   /* wLongsPerEntry */

            avio_w8(pb, 0);     /* bIndexSubType (0 == frame index) */

            avio_w8(pb, 0);     /* bIndexType (0 == AVI_INDEX_OF_INDEXES) */

            avio_wl32(pb, 0);   /* nEntriesInUse (will fill out later on) */

            ffio_wfourcc(pb, avi_stream2fourcc(tag, i, stream->codec_type));

                                /* dwChunkId */

            avio_wl64(pb, 0);   /* dwReserved[3] */

            // avio_wl32(pb, 0);   /* Must be 0.    */

            for (j = 0; j < AVI_MASTER_INDEX_SIZE * 2; j++)

                avio_wl64(pb, 0);

            ff_end_tag(pb, avist->indexes.indx_start);

        }



        if (stream->codec_type == AVMEDIA_TYPE_VIDEO   &&

            s->streams[i]->sample_aspect_ratio.num > 0 &&

            s->streams[i]->sample_aspect_ratio.den > 0) {

            int vprp       = ff_start_tag(pb, "vprp");

            AVRational dar = av_mul_q(s->streams[i]->sample_aspect_ratio,

                                      (AVRational) { stream->width,

                                                     stream->height });

            int num, den;

            av_reduce(&num, &den, dar.num, dar.den, 0xFFFF);



            avio_wl32(pb, 0); // video format   = unknown

            avio_wl32(pb, 0); // video standard = unknown

            avio_wl32(pb, lrintf(1.0 / av_q2d(stream->time_base)));

            avio_wl32(pb, stream->width);

            avio_wl32(pb, stream->height);

            avio_wl16(pb, den);

            avio_wl16(pb, num);

            avio_wl32(pb, stream->width);

            avio_wl32(pb, stream->height);

            avio_wl32(pb, 1); // progressive FIXME



            avio_wl32(pb, stream->height);

            avio_wl32(pb, stream->width);

            avio_wl32(pb, stream->height);

            avio_wl32(pb, stream->width);

            avio_wl32(pb, 0);

            avio_wl32(pb, 0);



            avio_wl32(pb, 0);

            avio_wl32(pb, 0);

            ff_end_tag(pb, vprp);

        }



        ff_end_tag(pb, list2);

    }



    if (pb->seekable) {

        /* AVI could become an OpenDML one, if it grows beyond 2Gb range */

        avi->odml_list = ff_start_tag(pb, "JUNK");

        ffio_wfourcc(pb, "odml");

        ffio_wfourcc(pb, "dmlh");

        avio_wl32(pb, 248);

        for (i = 0; i < 248; i += 4)

            avio_wl32(pb, 0);

        ff_end_tag(pb, avi->odml_list);

    }



    ff_end_tag(pb, list1);



    ff_riff_write_info(s);



    /* some padding for easier tag editing */

    list2 = ff_start_tag(pb, "JUNK");

    for (i = 0; i < 1016; i += 4)

        avio_wl32(pb, 0);

    ff_end_tag(pb, list2);



    avi->movi_list = ff_start_tag(pb, "LIST");

    ffio_wfourcc(pb, "movi");



    avio_flush(pb);



    return 0;

}
