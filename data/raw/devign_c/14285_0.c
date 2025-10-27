static int asf_write_header1(AVFormatContext *s, int64_t file_size, int64_t data_chunk_size)

{

    ASFContext *asf = s->priv_data;

    AVIOContext *pb = s->pb;

    AVDictionaryEntry *tags[5];

    int header_size, n, extra_size, extra_size2, wav_extra_size, file_time;

    int has_title;

    int metadata_count;

    AVCodecContext *enc;

    int64_t header_offset, cur_pos, hpos;

    int bit_rate;

    int64_t duration;



    ff_metadata_conv(&s->metadata, ff_asf_metadata_conv, NULL);



    tags[0] = av_dict_get(s->metadata, "title"    , NULL, 0);

    tags[1] = av_dict_get(s->metadata, "author"   , NULL, 0);

    tags[2] = av_dict_get(s->metadata, "copyright", NULL, 0);

    tags[3] = av_dict_get(s->metadata, "comment"  , NULL, 0);

    tags[4] = av_dict_get(s->metadata, "rating"   , NULL, 0);



    duration = asf->duration + PREROLL_TIME * 10000;

    has_title = tags[0] || tags[1] || tags[2] || tags[3] || tags[4];

    metadata_count = s->metadata ? s->metadata->count : 0;



    bit_rate = 0;

    for(n=0;n<s->nb_streams;n++) {

        enc = s->streams[n]->codec;



        av_set_pts_info(s->streams[n], 32, 1, 1000); /* 32 bit pts in ms */



        bit_rate += enc->bit_rate;

    }



    if (asf->is_streamed) {

        put_chunk(s, 0x4824, 0, 0xc00); /* start of stream (length will be patched later) */

    }



    put_guid(pb, &ff_asf_header);

    avio_wl64(pb, -1); /* header length, will be patched after */

    avio_wl32(pb, 3 + has_title + !!metadata_count + s->nb_streams); /* number of chunks in header */

    avio_w8(pb, 1); /* ??? */

    avio_w8(pb, 2); /* ??? */



    /* file header */

    header_offset = avio_tell(pb);

    hpos = put_header(pb, &ff_asf_file_header);

    put_guid(pb, &ff_asf_my_guid);

    avio_wl64(pb, file_size);

    file_time = 0;

    avio_wl64(pb, unix_to_file_time(file_time));

    avio_wl64(pb, asf->nb_packets); /* number of packets */

    avio_wl64(pb, duration); /* end time stamp (in 100ns units) */

    avio_wl64(pb, asf->duration); /* duration (in 100ns units) */

    avio_wl64(pb, PREROLL_TIME); /* start time stamp */

    avio_wl32(pb, (asf->is_streamed || !pb->seekable ) ? 3 : 2); /* ??? */

    avio_wl32(pb, s->packet_size); /* packet size */

    avio_wl32(pb, s->packet_size); /* packet size */

    avio_wl32(pb, bit_rate); /* Nominal data rate in bps */

    end_header(pb, hpos);



    /* unknown headers */

    hpos = put_header(pb, &ff_asf_head1_guid);

    put_guid(pb, &ff_asf_head2_guid);

    avio_wl32(pb, 6);

    avio_wl16(pb, 0);

    end_header(pb, hpos);



    /* title and other infos */

    if (has_title) {

        int len;

        uint8_t *buf;

        AVIOContext *dyn_buf;



        if (avio_open_dyn_buf(&dyn_buf) < 0)

            return AVERROR(ENOMEM);



        hpos = put_header(pb, &ff_asf_comment_header);



        for (n = 0; n < FF_ARRAY_ELEMS(tags); n++) {

            len = tags[n] ? avio_put_str16le(dyn_buf, tags[n]->value) : 0;

            avio_wl16(pb, len);

        }

        len = avio_close_dyn_buf(dyn_buf, &buf);

        avio_write(pb, buf, len);

        av_freep(&buf);

        end_header(pb, hpos);

    }

    if (metadata_count) {

        AVDictionaryEntry *tag = NULL;

        hpos = put_header(pb, &ff_asf_extended_content_header);

        avio_wl16(pb, metadata_count);

        while ((tag = av_dict_get(s->metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {

            put_str16(pb, tag->key);

            avio_wl16(pb, 0);

            put_str16(pb, tag->value);

        }

        end_header(pb, hpos);

    }



    /* stream headers */

    for(n=0;n<s->nb_streams;n++) {

        int64_t es_pos;

        //        ASFStream *stream = &asf->streams[n];



        enc = s->streams[n]->codec;

        asf->streams[n].num = n + 1;

        asf->streams[n].seq = 0;





        switch(enc->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            wav_extra_size = 0;

            extra_size = 18 + wav_extra_size;

            extra_size2 = 8;

            break;

        default:

        case AVMEDIA_TYPE_VIDEO:

            wav_extra_size = enc->extradata_size;

            extra_size = 0x33 + wav_extra_size;

            extra_size2 = 0;

            break;

        }



        hpos = put_header(pb, &ff_asf_stream_header);

        if (enc->codec_type == AVMEDIA_TYPE_AUDIO) {

            put_guid(pb, &ff_asf_audio_stream);

            put_guid(pb, &ff_asf_audio_conceal_spread);

        } else {

            put_guid(pb, &ff_asf_video_stream);

            put_guid(pb, &ff_asf_video_conceal_none);

        }

        avio_wl64(pb, 0); /* ??? */

        es_pos = avio_tell(pb);

        avio_wl32(pb, extra_size); /* wav header len */

        avio_wl32(pb, extra_size2); /* additional data len */

        avio_wl16(pb, n + 1); /* stream number */

        avio_wl32(pb, 0); /* ??? */



        if (enc->codec_type == AVMEDIA_TYPE_AUDIO) {

            /* WAVEFORMATEX header */

            int wavsize = ff_put_wav_header(pb, enc);

            if ((enc->codec_id != CODEC_ID_MP3) && (enc->codec_id != CODEC_ID_MP2) && (enc->codec_id != CODEC_ID_ADPCM_IMA_WAV) && (enc->extradata_size==0)) {

                wavsize += 2;

                avio_wl16(pb, 0);

            }



            if (wavsize < 0)

                return -1;

            if (wavsize != extra_size) {

                cur_pos = avio_tell(pb);

                avio_seek(pb, es_pos, SEEK_SET);

                avio_wl32(pb, wavsize); /* wav header len */

                avio_seek(pb, cur_pos, SEEK_SET);

            }

            /* ERROR Correction */

            avio_w8(pb, 0x01);

            if(enc->codec_id == CODEC_ID_ADPCM_G726 || !enc->block_align){

                avio_wl16(pb, 0x0190);

                avio_wl16(pb, 0x0190);

            }else{

                avio_wl16(pb, enc->block_align);

                avio_wl16(pb, enc->block_align);

            }

            avio_wl16(pb, 0x01);

            avio_w8(pb, 0x00);

        } else {

            avio_wl32(pb, enc->width);

            avio_wl32(pb, enc->height);

            avio_w8(pb, 2); /* ??? */

            avio_wl16(pb, 40 + enc->extradata_size); /* size */



            /* BITMAPINFOHEADER header */

            ff_put_bmp_header(pb, enc, ff_codec_bmp_tags, 1);

        }

        end_header(pb, hpos);

    }



    /* media comments */



    hpos = put_header(pb, &ff_asf_codec_comment_header);

    put_guid(pb, &ff_asf_codec_comment1_header);

    avio_wl32(pb, s->nb_streams);

    for(n=0;n<s->nb_streams;n++) {

        AVCodec *p;

        const char *desc;

        int len;

        uint8_t *buf;

        AVIOContext *dyn_buf;



        enc = s->streams[n]->codec;

        p = avcodec_find_encoder(enc->codec_id);



        if(enc->codec_type == AVMEDIA_TYPE_AUDIO)

            avio_wl16(pb, 2);

        else if(enc->codec_type == AVMEDIA_TYPE_VIDEO)

            avio_wl16(pb, 1);

        else

            avio_wl16(pb, -1);



        if(enc->codec_id == CODEC_ID_WMAV2)

            desc = "Windows Media Audio V8";

        else

            desc = p ? p->name : enc->codec_name;



        if ( avio_open_dyn_buf(&dyn_buf) < 0)

            return AVERROR(ENOMEM);



        avio_put_str16le(dyn_buf, desc);

        len = avio_close_dyn_buf(dyn_buf, &buf);

        avio_wl16(pb, len / 2); // "number of characters" = length in bytes / 2



        avio_write(pb, buf, len);

        av_freep(&buf);



        avio_wl16(pb, 0); /* no parameters */





        /* id */

        if (enc->codec_type == AVMEDIA_TYPE_AUDIO) {

            avio_wl16(pb, 2);

            avio_wl16(pb, enc->codec_tag);

        } else {

            avio_wl16(pb, 4);

            avio_wl32(pb, enc->codec_tag);

        }

        if(!enc->codec_tag)

            return -1;

    }

    end_header(pb, hpos);



    /* patch the header size fields */



    cur_pos = avio_tell(pb);

    header_size = cur_pos - header_offset;

    if (asf->is_streamed) {

        header_size += 8 + 30 + 50;



        avio_seek(pb, header_offset - 10 - 30, SEEK_SET);

        avio_wl16(pb, header_size);

        avio_seek(pb, header_offset - 2 - 30, SEEK_SET);

        avio_wl16(pb, header_size);



        header_size -= 8 + 30 + 50;

    }

    header_size += 24 + 6;

    avio_seek(pb, header_offset - 14, SEEK_SET);

    avio_wl64(pb, header_size);

    avio_seek(pb, cur_pos, SEEK_SET);



    /* movie chunk, followed by packets of packet_size */

    asf->data_offset = cur_pos;

    put_guid(pb, &ff_asf_data_header);

    avio_wl64(pb, data_chunk_size);

    put_guid(pb, &ff_asf_my_guid);

    avio_wl64(pb, asf->nb_packets); /* nb packets */

    avio_w8(pb, 1); /* ??? */

    avio_w8(pb, 1); /* ??? */

    return 0;

}
