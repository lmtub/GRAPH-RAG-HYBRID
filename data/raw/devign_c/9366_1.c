static int parse_chunks(AVFormatContext *s, int mode, int64_t seekts, int *len_ptr)

{

    WtvContext *wtv = s->priv_data;

    AVIOContext *pb = wtv->pb;

    while (!url_feof(pb)) {

        ff_asf_guid g;

        int len, sid, consumed;



        ff_get_guid(pb, &g);

        len = avio_rl32(pb);

        if (len < 32)

            break;

        sid = avio_rl32(pb) & 0x7FFF;

        avio_skip(pb, 8);

        consumed = 32;



        if (!ff_guidcmp(g, ff_SBE2_STREAM_DESC_EVENT)) {

            if (ff_find_stream_index(s, sid) < 0) {

                ff_asf_guid mediatype, subtype, formattype;

                int size;

                avio_skip(pb, 28);

                ff_get_guid(pb, &mediatype);

                ff_get_guid(pb, &subtype);

                avio_skip(pb, 12);

                ff_get_guid(pb, &formattype);

                size = avio_rl32(pb);

                parse_media_type(s, 0, sid, mediatype, subtype, formattype, size);

                consumed += 92 + size;

            }

        } else if (!ff_guidcmp(g, ff_stream2_guid)) {

            int stream_index = ff_find_stream_index(s, sid);

            if (stream_index >= 0 && !((WtvStream*)s->streams[stream_index]->priv_data)->seen_data) {

                ff_asf_guid mediatype, subtype, formattype;

                int size;

                avio_skip(pb, 12);

                ff_get_guid(pb, &mediatype);

                ff_get_guid(pb, &subtype);

                avio_skip(pb, 12);

                ff_get_guid(pb, &formattype);

                size = avio_rl32(pb);

                parse_media_type(s, s->streams[stream_index], sid, mediatype, subtype, formattype, size);

                consumed += 76 + size;

            }

        } else if (!ff_guidcmp(g, EVENTID_AudioDescriptorSpanningEvent) ||

                   !ff_guidcmp(g, EVENTID_CtxADescriptorSpanningEvent) ||

                   !ff_guidcmp(g, EVENTID_CSDescriptorSpanningEvent) ||

                   !ff_guidcmp(g, EVENTID_StreamIDSpanningEvent) ||

                   !ff_guidcmp(g, EVENTID_SubtitleSpanningEvent) ||

                   !ff_guidcmp(g, EVENTID_TeletextSpanningEvent)) {

            int stream_index = ff_find_stream_index(s, sid);

            if (stream_index >= 0) {

                AVStream *st = s->streams[stream_index];

                uint8_t buf[258];

                const uint8_t *pbuf = buf;

                int buf_size;



                avio_skip(pb, 8);

                consumed += 8;

                if (!ff_guidcmp(g, EVENTID_CtxADescriptorSpanningEvent) ||

                    !ff_guidcmp(g, EVENTID_CSDescriptorSpanningEvent)) {

                    avio_skip(pb, 6);

                    consumed += 6;

                }



                buf_size = FFMIN(len - consumed, sizeof(buf));

                avio_read(pb, buf, buf_size);

                consumed += buf_size;

                ff_parse_mpeg2_descriptor(s, st, 0, &pbuf, buf + buf_size, NULL, 0, 0, NULL);

            }

        } else if (!ff_guidcmp(g, EVENTID_AudioTypeSpanningEvent)) {

            int stream_index = ff_find_stream_index(s, sid);

            if (stream_index >= 0) {

                AVStream *st = s->streams[stream_index];

                int audio_type;

                avio_skip(pb, 8);

                audio_type = avio_r8(pb);

                if (audio_type == 2)

                    st->disposition |= AV_DISPOSITION_HEARING_IMPAIRED;

                else if (audio_type == 3)

                    st->disposition |= AV_DISPOSITION_VISUAL_IMPAIRED;

                consumed += 9;

            }

        } else if (!ff_guidcmp(g, EVENTID_DVBScramblingControlSpanningEvent)) {

            int stream_index = ff_find_stream_index(s, sid);

            if (stream_index >= 0) {

                avio_skip(pb, 12);

                if (avio_rl32(pb))

                    av_log(s, AV_LOG_WARNING, "DVB scrambled stream detected (st:%d), decoding will likely fail\n", stream_index);

                consumed += 16;

            }

        } else if (!ff_guidcmp(g, EVENTID_LanguageSpanningEvent)) {

            int stream_index = ff_find_stream_index(s, sid);

            if (stream_index >= 0) {

                AVStream *st = s->streams[stream_index];

                uint8_t language[4];

                avio_skip(pb, 12);

                avio_read(pb, language, 3);

                if (language[0]) {

                    language[3] = 0;

                    av_dict_set(&st->metadata, "language", language, 0);

                    if (!strcmp(language, "nar") || !strcmp(language, "NAR"))

                        st->disposition |= AV_DISPOSITION_VISUAL_IMPAIRED;

                }

                consumed += 15;

            }

        } else if (!ff_guidcmp(g, ff_timestamp_guid)) {

            int stream_index = ff_find_stream_index(s, sid);

            if (stream_index >= 0) {

                avio_skip(pb, 8);

                wtv->pts = avio_rl64(pb);

                consumed += 16;

                if (wtv->pts == -1)

                    wtv->pts = AV_NOPTS_VALUE;

                else {

                    wtv->last_valid_pts = wtv->pts;

                    if (wtv->epoch == AV_NOPTS_VALUE || wtv->pts < wtv->epoch)

                        wtv->epoch = wtv->pts;

                if (mode == SEEK_TO_PTS && wtv->pts >= seekts) {

                    avio_skip(pb, WTV_PAD8(len) - consumed);

                    return 0;

                }

                }

            }

        } else if (!ff_guidcmp(g, ff_data_guid)) {

            int stream_index = ff_find_stream_index(s, sid);

            if (mode == SEEK_TO_DATA && stream_index >= 0 && len > 32 && s->streams[stream_index]->priv_data) {

                WtvStream *wst = s->streams[stream_index]->priv_data;

                wst->seen_data = 1;

                if (len_ptr) {

                    *len_ptr = len;

                }

                return stream_index;

            }

        } else if (!ff_guidcmp(g, /* DSATTRIB_WMDRMProtectionInfo */ (const ff_asf_guid){0x83,0x95,0x74,0x40,0x9D,0x6B,0xEC,0x4E,0xB4,0x3C,0x67,0xA1,0x80,0x1E,0x1A,0x9B})) {

            int stream_index = ff_find_stream_index(s, sid);

            if (stream_index >= 0)

                av_log(s, AV_LOG_WARNING, "encrypted stream detected (st:%d), decoding will likely fail\n", stream_index);

        } else if (

            !ff_guidcmp(g, /* DSATTRIB_CAPTURE_STREAMTIME */ (const ff_asf_guid){0x14,0x56,0x1A,0x0C,0xCD,0x30,0x40,0x4F,0xBC,0xBF,0xD0,0x3E,0x52,0x30,0x62,0x07}) ||

            !ff_guidcmp(g, /* DSATTRIB_PBDATAG_ATTRIBUTE */ (const ff_asf_guid){0x79,0x66,0xB5,0xE0,0xB9,0x12,0xCC,0x43,0xB7,0xDF,0x57,0x8C,0xAA,0x5A,0x7B,0x63}) ||

            !ff_guidcmp(g, /* DSATTRIB_PicSampleSeq */ (const ff_asf_guid){0x02,0xAE,0x5B,0x2F,0x8F,0x7B,0x60,0x4F,0x82,0xD6,0xE4,0xEA,0x2F,0x1F,0x4C,0x99}) ||

            !ff_guidcmp(g, /* DSATTRIB_TRANSPORT_PROPERTIES */ ff_DSATTRIB_TRANSPORT_PROPERTIES) ||

            !ff_guidcmp(g, /* dvr_ms_vid_frame_rep_data */ (const ff_asf_guid){0xCC,0x32,0x64,0xDD,0x29,0xE2,0xDB,0x40,0x80,0xF6,0xD2,0x63,0x28,0xD2,0x76,0x1F}) ||

            !ff_guidcmp(g, /* EVENTID_ChannelChangeSpanningEvent */ (const ff_asf_guid){0xE5,0xC5,0x67,0x90,0x5C,0x4C,0x05,0x42,0x86,0xC8,0x7A,0xFE,0x20,0xFE,0x1E,0xFA}) ||

            !ff_guidcmp(g, /* EVENTID_ChannelInfoSpanningEvent */ (const ff_asf_guid){0x80,0x6D,0xF3,0x41,0x32,0x41,0xC2,0x4C,0xB1,0x21,0x01,0xA4,0x32,0x19,0xD8,0x1B}) ||

            !ff_guidcmp(g, /* EVENTID_ChannelTypeSpanningEvent */ (const ff_asf_guid){0x51,0x1D,0xAB,0x72,0xD2,0x87,0x9B,0x48,0xBA,0x11,0x0E,0x08,0xDC,0x21,0x02,0x43}) ||

            !ff_guidcmp(g, /* EVENTID_PIDListSpanningEvent */ (const ff_asf_guid){0x65,0x8F,0xFC,0x47,0xBB,0xE2,0x34,0x46,0x9C,0xEF,0xFD,0xBF,0xE6,0x26,0x1D,0x5C}) ||

            !ff_guidcmp(g, /* EVENTID_SignalAndServiceStatusSpanningEvent */ (const ff_asf_guid){0xCB,0xC5,0x68,0x80,0x04,0x3C,0x2B,0x49,0xB4,0x7D,0x03,0x08,0x82,0x0D,0xCE,0x51}) ||

            !ff_guidcmp(g, /* EVENTID_StreamTypeSpanningEvent */ (const ff_asf_guid){0xBC,0x2E,0xAF,0x82,0xA6,0x30,0x64,0x42,0xA8,0x0B,0xAD,0x2E,0x13,0x72,0xAC,0x60}) ||

            !ff_guidcmp(g, (const ff_asf_guid){0x1E,0xBE,0xC3,0xC5,0x43,0x92,0xDC,0x11,0x85,0xE5,0x00,0x12,0x3F,0x6F,0x73,0xB9}) ||

            !ff_guidcmp(g, (const ff_asf_guid){0x3B,0x86,0xA2,0xB1,0xEB,0x1E,0xC3,0x44,0x8C,0x88,0x1C,0xA3,0xFF,0xE3,0xE7,0x6A}) ||

            !ff_guidcmp(g, (const ff_asf_guid){0x4E,0x7F,0x4C,0x5B,0xC4,0xD0,0x38,0x4B,0xA8,0x3E,0x21,0x7F,0x7B,0xBF,0x52,0xE7}) ||

            !ff_guidcmp(g, (const ff_asf_guid){0x63,0x36,0xEB,0xFE,0xA1,0x7E,0xD9,0x11,0x83,0x08,0x00,0x07,0xE9,0x5E,0xAD,0x8D}) ||

            !ff_guidcmp(g, (const ff_asf_guid){0x70,0xE9,0xF1,0xF8,0x89,0xA4,0x4C,0x4D,0x83,0x73,0xB8,0x12,0xE0,0xD5,0xF8,0x1E}) ||

            !ff_guidcmp(g, ff_index_guid) ||

            !ff_guidcmp(g, ff_sync_guid) ||

            !ff_guidcmp(g, ff_stream1_guid) ||

            !ff_guidcmp(g, (const ff_asf_guid){0xF7,0x10,0x02,0xB9,0xEE,0x7C,0xED,0x4E,0xBD,0x7F,0x05,0x40,0x35,0x86,0x18,0xA1})) {

            //ignore known guids

        } else

            av_log(s, AV_LOG_WARNING, "unsupported chunk:"FF_PRI_GUID"\n", FF_ARG_GUID(g));



        avio_skip(pb, WTV_PAD8(len) - consumed);

    }

    return AVERROR_EOF;

}
