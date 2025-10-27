static int asf_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    ASFContext *asf = s->priv_data;

    GUID g;

    ByteIOContext *pb = &s->pb;

    AVStream *st;

    ASFStream *asf_st;

    int size, i, bps;

    INT64 gsize;



    get_guid(pb, &g);

    if (memcmp(&g, &asf_header, sizeof(GUID)))

        goto fail;

    get_le64(pb);

    get_le32(pb);

    get_byte(pb);

    get_byte(pb);

    memset(&asf->asfid2avid, -1, sizeof(asf->asfid2avid));

    for(;;) {

        get_guid(pb, &g);

        gsize = get_le64(pb);

#ifdef DEBUG

        printf("%08Lx: ", url_ftell(pb) - 24);

        print_guid(&g);

        printf("  size=0x%Lx\n", gsize);

#endif

        if (gsize < 24)

            goto fail;

        if (!memcmp(&g, &file_header, sizeof(GUID))) {

            get_guid(pb, &asf->hdr.guid);

	    asf->hdr.file_size		= get_le64(pb);

	    asf->hdr.create_time	= get_le64(pb);

	    asf->hdr.packets_count	= get_le64(pb);

	    asf->hdr.play_time		= get_le64(pb);

	    asf->hdr.send_time		= get_le64(pb);

	    asf->hdr.preroll		= get_le32(pb);

	    asf->hdr.ignore		= get_le32(pb);

	    asf->hdr.flags		= get_le32(pb);

	    asf->hdr.min_pktsize	= get_le32(pb);

	    asf->hdr.max_pktsize	= get_le32(pb);

	    asf->hdr.max_bitrate	= get_le32(pb);

	    asf->packet_size = asf->hdr.max_pktsize;

            asf->nb_packets = asf->hdr.packets_count;

        } else if (!memcmp(&g, &stream_header, sizeof(GUID))) {

            int type, id, total_size;

            unsigned int tag1;

            INT64 pos1, pos2;



            pos1 = url_ftell(pb);



            st = av_mallocz(sizeof(AVStream));

            if (!st)

                goto fail;

            s->streams[s->nb_streams] = st;

            asf_st = av_mallocz(sizeof(ASFStream));

            if (!asf_st)

                goto fail;

            st->priv_data = asf_st;

	    st->time_length = (asf->hdr.send_time - asf->hdr.preroll) / 10000;

            get_guid(pb, &g);

            if (!memcmp(&g, &audio_stream, sizeof(GUID))) {

                type = CODEC_TYPE_AUDIO;

            } else if (!memcmp(&g, &video_stream, sizeof(GUID))) {

                type = CODEC_TYPE_VIDEO;

            } else {

                goto fail;

            }

            get_guid(pb, &g);

            total_size = get_le64(pb);

            get_le32(pb);

            get_le32(pb);

	    st->id = get_le16(pb) & 0x7f; /* stream id */

            // mapping of asf ID to AV stream ID;

            asf->asfid2avid[st->id] = s->nb_streams++;



            get_le32(pb);

	    st->codec.codec_type = type;

            st->codec.frame_rate = 1000; // in packet ticks

            if (type == CODEC_TYPE_AUDIO) {

                id = get_le16(pb);

                st->codec.codec_tag = id;

                st->codec.channels = get_le16(pb);

		st->codec.sample_rate = get_le32(pb);

                st->codec.bit_rate = get_le32(pb) * 8;

		st->codec.block_align = get_le16(pb); /* block align */

                bps = get_le16(pb); /* bits per sample */

                st->codec.codec_id = wav_codec_get_id(id, bps);

		size = get_le16(pb);

		if (size > 0) {

		    st->extra_data = av_mallocz(size);

		    get_buffer(pb, st->extra_data, size);

		    st->extra_data_size = size;

		}

		/* We have to init the frame size at some point .... */

		pos2 = url_ftell(pb);

		if (gsize > (pos2 + 8 - pos1 + 24)) {

		    asf_st->ds_span = get_byte(pb);

		    asf_st->ds_packet_size = get_le16(pb);

		    asf_st->ds_chunk_size = get_le16(pb);

		    asf_st->ds_data_size = get_le16(pb);

		    asf_st->ds_silence_data = get_byte(pb);

		}

		//printf("Descrambling: ps:%d cs:%d ds:%d s:%d  sd:%d\n",

		//       asf_st->ds_packet_size, asf_st->ds_chunk_size,

		//       asf_st->ds_data_size, asf_st->ds_span, asf_st->ds_silence_data);

		if (asf_st->ds_span > 1) {

		    if (!asf_st->ds_chunk_size

			|| (asf_st->ds_packet_size/asf_st->ds_chunk_size <= 1))

			asf_st->ds_span = 0; // disable descrambling

		}

                switch (st->codec.codec_id) {

                case CODEC_ID_MP3LAME:

                    st->codec.frame_size = MPA_FRAME_SIZE;

                    break;

                case CODEC_ID_PCM_S16LE:

                case CODEC_ID_PCM_S16BE:

                case CODEC_ID_PCM_U16LE:

                case CODEC_ID_PCM_U16BE:

                case CODEC_ID_PCM_S8:

                case CODEC_ID_PCM_U8:

                case CODEC_ID_PCM_ALAW:

                case CODEC_ID_PCM_MULAW:

                    st->codec.frame_size = 1;

                    break;

                default:

                    /* This is probably wrong, but it prevents a crash later */

                    st->codec.frame_size = 1;

                    break;

                }

            } else {

		get_le32(pb);

                get_le32(pb);

                get_byte(pb);

                size = get_le16(pb); /* size */

                get_le32(pb); /* size */

                st->codec.width = get_le32(pb);

		st->codec.height = get_le32(pb);

                /* not available for asf */

                get_le16(pb); /* panes */

                get_le16(pb); /* depth */

                tag1 = get_le32(pb);

		url_fskip(pb, 20);

		if (size > 40) {

		    st->extra_data_size = size - 40;

		    st->extra_data = av_mallocz(st->extra_data_size);

		    get_buffer(pb, st->extra_data, st->extra_data_size);

		}

                st->codec.codec_tag = tag1;

		st->codec.codec_id = codec_get_id(codec_asf_bmp_tags, tag1);

            }

            pos2 = url_ftell(pb);

            url_fskip(pb, gsize - (pos2 - pos1 + 24));

        } else if (!memcmp(&g, &data_header, sizeof(GUID))) {

            break;

        } else if (!memcmp(&g, &comment_header, sizeof(GUID))) {

            int len1, len2, len3, len4, len5;



            len1 = get_le16(pb);

            len2 = get_le16(pb);

            len3 = get_le16(pb);

            len4 = get_le16(pb);

            len5 = get_le16(pb);

            get_str16_nolen(pb, len1, s->title, sizeof(s->title));

            get_str16_nolen(pb, len2, s->author, sizeof(s->author));

            get_str16_nolen(pb, len3, s->copyright, sizeof(s->copyright));

            get_str16_nolen(pb, len4, s->comment, sizeof(s->comment));

	    url_fskip(pb, len5);

#if 0

        } else if (!memcmp(&g, &head1_guid, sizeof(GUID))) {

            int v1, v2;

            get_guid(pb, &g);

            v1 = get_le32(pb);

            v2 = get_le16(pb);

        } else if (!memcmp(&g, &codec_comment_header, sizeof(GUID))) {

            int len, v1, n, num;

            char str[256], *q;

            char tag[16];



            get_guid(pb, &g);

            print_guid(&g);



            n = get_le32(pb);

            for(i=0;i<n;i++) {

                num = get_le16(pb); /* stream number */

                get_str16(pb, str, sizeof(str));

                get_str16(pb, str, sizeof(str));

                len = get_le16(pb);

                q = tag;

                while (len > 0) {

                    v1 = get_byte(pb);

                    if ((q - tag) < sizeof(tag) - 1)

                        *q++ = v1;

                    len--;

                }

                *q = '\0';

            }

#endif

        } else if (url_feof(pb)) {

            goto fail;

        } else {

            url_fseek(pb, gsize - 24, SEEK_CUR);

        }

    }

    get_guid(pb, &g);

    get_le64(pb);

    get_byte(pb);

    get_byte(pb);

    if (url_feof(pb))

        goto fail;

    asf->data_offset = url_ftell(pb);

    asf->packet_size_left = 0;



    return 0;



 fail:

    for(i=0;i<s->nb_streams;i++) {

        AVStream *st = s->streams[i];

	if (st) {

	    av_free(st->priv_data);

	    av_free(st->extra_data);

	}

        av_free(st);

    }

    //av_free(asf);

    return -1;

}
