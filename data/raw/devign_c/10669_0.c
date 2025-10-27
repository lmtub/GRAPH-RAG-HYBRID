static int mov_init(AVFormatContext *s)

{

    MOVMuxContext *mov = s->priv_data;

    AVDictionaryEntry *global_tcr = av_dict_get(s->metadata, "timecode", NULL, 0);

    int i, ret;



    mov->fc = s;



    /* Default mode == MP4 */

    mov->mode = MODE_MP4;



    if (s->oformat) {

        if (!strcmp("3gp", s->oformat->name)) mov->mode = MODE_3GP;

        else if (!strcmp("3g2", s->oformat->name)) mov->mode = MODE_3GP|MODE_3G2;

        else if (!strcmp("mov", s->oformat->name)) mov->mode = MODE_MOV;

        else if (!strcmp("psp", s->oformat->name)) mov->mode = MODE_PSP;

        else if (!strcmp("ipod",s->oformat->name)) mov->mode = MODE_IPOD;

        else if (!strcmp("ismv",s->oformat->name)) mov->mode = MODE_ISM;

        else if (!strcmp("f4v", s->oformat->name)) mov->mode = MODE_F4V;

    }



    if (mov->flags & FF_MOV_FLAG_DELAY_MOOV)

        mov->flags |= FF_MOV_FLAG_EMPTY_MOOV;



    /* Set the FRAGMENT flag if any of the fragmentation methods are

     * enabled. */

    if (mov->max_fragment_duration || mov->max_fragment_size ||

        mov->flags & (FF_MOV_FLAG_EMPTY_MOOV |

                      FF_MOV_FLAG_FRAG_KEYFRAME |

                      FF_MOV_FLAG_FRAG_CUSTOM))

        mov->flags |= FF_MOV_FLAG_FRAGMENT;



    /* Set other implicit flags immediately */

    if (mov->mode == MODE_ISM)

        mov->flags |= FF_MOV_FLAG_EMPTY_MOOV | FF_MOV_FLAG_SEPARATE_MOOF |

                      FF_MOV_FLAG_FRAGMENT;

    if (mov->flags & FF_MOV_FLAG_DASH)

        mov->flags |= FF_MOV_FLAG_FRAGMENT | FF_MOV_FLAG_EMPTY_MOOV |

                      FF_MOV_FLAG_DEFAULT_BASE_MOOF;



    if (mov->flags & FF_MOV_FLAG_EMPTY_MOOV && s->flags & AVFMT_FLAG_AUTO_BSF) {

        av_log(s, AV_LOG_VERBOSE, "Empty MOOV enabled; disabling automatic bitstream filtering\n");

        s->flags &= ~AVFMT_FLAG_AUTO_BSF;

    }



    if (mov->flags & FF_MOV_FLAG_FASTSTART) {

        mov->reserved_moov_size = -1;

    }



    if (mov->use_editlist < 0) {

        mov->use_editlist = 1;

        if (mov->flags & FF_MOV_FLAG_FRAGMENT &&

            !(mov->flags & FF_MOV_FLAG_DELAY_MOOV)) {

            // If we can avoid needing an edit list by shifting the

            // tracks, prefer that over (trying to) write edit lists

            // in fragmented output.

            if (s->avoid_negative_ts == AVFMT_AVOID_NEG_TS_AUTO ||

                s->avoid_negative_ts == AVFMT_AVOID_NEG_TS_MAKE_ZERO)

                mov->use_editlist = 0;

        }

    }

    if (mov->flags & FF_MOV_FLAG_EMPTY_MOOV &&

        !(mov->flags & FF_MOV_FLAG_DELAY_MOOV) && mov->use_editlist)

        av_log(s, AV_LOG_WARNING, "No meaningful edit list will be written when using empty_moov without delay_moov\n");



    if (!mov->use_editlist && s->avoid_negative_ts == AVFMT_AVOID_NEG_TS_AUTO)

        s->avoid_negative_ts = AVFMT_AVOID_NEG_TS_MAKE_ZERO;



    /* Clear the omit_tfhd_offset flag if default_base_moof is set;

     * if the latter is set that's enough and omit_tfhd_offset doesn't

     * add anything extra on top of that. */

    if (mov->flags & FF_MOV_FLAG_OMIT_TFHD_OFFSET &&

        mov->flags & FF_MOV_FLAG_DEFAULT_BASE_MOOF)

        mov->flags &= ~FF_MOV_FLAG_OMIT_TFHD_OFFSET;



    if (mov->frag_interleave &&

        mov->flags & (FF_MOV_FLAG_OMIT_TFHD_OFFSET | FF_MOV_FLAG_SEPARATE_MOOF)) {

        av_log(s, AV_LOG_ERROR,

               "Sample interleaving in fragments is mutually exclusive with "

               "omit_tfhd_offset and separate_moof\n");

        return AVERROR(EINVAL);

    }



    /* Non-seekable output is ok if using fragmentation. If ism_lookahead

     * is enabled, we don't support non-seekable output at all. */

    if (!(s->pb->seekable & AVIO_SEEKABLE_NORMAL) &&

        (!(mov->flags & FF_MOV_FLAG_FRAGMENT) || mov->ism_lookahead)) {

        av_log(s, AV_LOG_ERROR, "muxer does not support non seekable output\n");

        return AVERROR(EINVAL);

    }



    mov->nb_streams = s->nb_streams;

    if (mov->mode & (MODE_MP4|MODE_MOV|MODE_IPOD) && s->nb_chapters)

        mov->chapter_track = mov->nb_streams++;



    if (mov->flags & FF_MOV_FLAG_RTP_HINT) {

        /* Add hint tracks for each audio and video stream */

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ||

                st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {

                mov->nb_streams++;

            }

        }

    }



    if (   mov->write_tmcd == -1 && (mov->mode == MODE_MOV || mov->mode == MODE_MP4)

        || mov->write_tmcd == 1) {

        /* +1 tmcd track for each video stream with a timecode */

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            AVDictionaryEntry *t = global_tcr;

            if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO &&

                (t || (t=av_dict_get(st->metadata, "timecode", NULL, 0)))) {

                AVTimecode tc;

                ret = mov_check_timecode_track(s, &tc, i, t->value);

                if (ret >= 0)

                    mov->nb_meta_tmcd++;

            }

        }



        /* check if there is already a tmcd track to remux */

        if (mov->nb_meta_tmcd) {

            for (i = 0; i < s->nb_streams; i++) {

                AVStream *st = s->streams[i];

                if (st->codecpar->codec_tag == MKTAG('t','m','c','d')) {

                    av_log(s, AV_LOG_WARNING, "You requested a copy of the original timecode track "

                           "so timecode metadata are now ignored\n");

                    mov->nb_meta_tmcd = 0;

                }

            }

        }



        mov->nb_streams += mov->nb_meta_tmcd;

    }



    // Reserve an extra stream for chapters for the case where chapters

    // are written in the trailer

    mov->tracks = av_mallocz_array((mov->nb_streams + 1), sizeof(*mov->tracks));

    if (!mov->tracks)

        return AVERROR(ENOMEM);



    if (mov->encryption_scheme_str != NULL && strcmp(mov->encryption_scheme_str, "none") != 0) {

        if (strcmp(mov->encryption_scheme_str, "cenc-aes-ctr") == 0) {

            mov->encryption_scheme = MOV_ENC_CENC_AES_CTR;



            if (mov->encryption_key_len != AES_CTR_KEY_SIZE) {

                av_log(s, AV_LOG_ERROR, "Invalid encryption key len %d expected %d\n",

                    mov->encryption_key_len, AES_CTR_KEY_SIZE);

                return AVERROR(EINVAL);

            }



            if (mov->encryption_kid_len != CENC_KID_SIZE) {

                av_log(s, AV_LOG_ERROR, "Invalid encryption kid len %d expected %d\n",

                    mov->encryption_kid_len, CENC_KID_SIZE);

                return AVERROR(EINVAL);

            }

        } else {

            av_log(s, AV_LOG_ERROR, "unsupported encryption scheme %s\n",

                mov->encryption_scheme_str);

            return AVERROR(EINVAL);

        }

    }



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st= s->streams[i];

        MOVTrack *track= &mov->tracks[i];

        AVDictionaryEntry *lang = av_dict_get(st->metadata, "language", NULL,0);



        track->st  = st;

        track->par = st->codecpar;

        track->language = ff_mov_iso639_to_lang(lang?lang->value:"und", mov->mode!=MODE_MOV);

        if (track->language < 0)

            track->language = 0;

        track->mode = mov->mode;

        track->tag  = mov_find_codec_tag(s, track);

        if (!track->tag) {

            av_log(s, AV_LOG_ERROR, "Could not find tag for codec %s in stream #%d, "

                   "codec not currently supported in container\n",

                   avcodec_get_name(st->codecpar->codec_id), i);

            return AVERROR(EINVAL);

        }

        /* If hinting of this track is enabled by a later hint track,

         * this is updated. */

        track->hint_track = -1;

        track->start_dts  = AV_NOPTS_VALUE;

        track->start_cts  = AV_NOPTS_VALUE;

        track->end_pts    = AV_NOPTS_VALUE;

        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {

            if (track->tag == MKTAG('m','x','3','p') || track->tag == MKTAG('m','x','3','n') ||

                track->tag == MKTAG('m','x','4','p') || track->tag == MKTAG('m','x','4','n') ||

                track->tag == MKTAG('m','x','5','p') || track->tag == MKTAG('m','x','5','n')) {

                if (st->codecpar->width != 720 || (st->codecpar->height != 608 && st->codecpar->height != 512)) {

                    av_log(s, AV_LOG_ERROR, "D-10/IMX must use 720x608 or 720x512 video resolution\n");

                    return AVERROR(EINVAL);

                }

                track->height = track->tag >> 24 == 'n' ? 486 : 576;

            }

            if (mov->video_track_timescale) {

                track->timescale = mov->video_track_timescale;

            } else {

                track->timescale = st->time_base.den;

                while(track->timescale < 10000)

                    track->timescale *= 2;

            }

            if (st->codecpar->width > 65535 || st->codecpar->height > 65535) {

                av_log(s, AV_LOG_ERROR, "Resolution %dx%d too large for mov/mp4\n", st->codecpar->width, st->codecpar->height);

                return AVERROR(EINVAL);

            }

            if (track->mode == MODE_MOV && track->timescale > 100000)

                av_log(s, AV_LOG_WARNING,

                       "WARNING codec timebase is very high. If duration is too long,\n"

                       "file may not be playable by quicktime. Specify a shorter timebase\n"

                       "or choose different container.\n");

            if (track->mode == MODE_MOV &&

                track->par->codec_id == AV_CODEC_ID_RAWVIDEO &&

                track->tag == MKTAG('r','a','w',' ')) {

                enum AVPixelFormat pix_fmt = track->par->format;

                if (pix_fmt == AV_PIX_FMT_NONE && track->par->bits_per_coded_sample == 1)

                    pix_fmt = AV_PIX_FMT_MONOWHITE;

                track->is_unaligned_qt_rgb =

                        pix_fmt == AV_PIX_FMT_RGB24 ||

                        pix_fmt == AV_PIX_FMT_BGR24 ||

                        pix_fmt == AV_PIX_FMT_PAL8 ||

                        pix_fmt == AV_PIX_FMT_GRAY8 ||

                        pix_fmt == AV_PIX_FMT_MONOWHITE ||

                        pix_fmt == AV_PIX_FMT_MONOBLACK;

            }

            if (track->par->codec_id == AV_CODEC_ID_VP9) {

                if (track->mode != MODE_MP4) {

                    av_log(s, AV_LOG_ERROR, "VP9 only supported in MP4.\n");

                    return AVERROR(EINVAL);

                }

                if (s->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {

                    av_log(s, AV_LOG_ERROR,

                           "VP9 in MP4 support is experimental, add "

                           "'-strict %d' if you want to use it.\n",

                           FF_COMPLIANCE_EXPERIMENTAL);

                    return AVERROR_EXPERIMENTAL;

                }

            }

        } else if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {

            track->timescale = st->codecpar->sample_rate;

            if (!st->codecpar->frame_size && !av_get_bits_per_sample(st->codecpar->codec_id)) {

                av_log(s, AV_LOG_WARNING, "track %d: codec frame size is not set\n", i);

                track->audio_vbr = 1;

            }else if (st->codecpar->codec_id == AV_CODEC_ID_ADPCM_MS ||

                     st->codecpar->codec_id == AV_CODEC_ID_ADPCM_IMA_WAV ||

                     st->codecpar->codec_id == AV_CODEC_ID_ILBC){

                if (!st->codecpar->block_align) {

                    av_log(s, AV_LOG_ERROR, "track %d: codec block align is not set for adpcm\n", i);

                    return AVERROR(EINVAL);

                }

                track->sample_size = st->codecpar->block_align;

            }else if (st->codecpar->frame_size > 1){ /* assume compressed audio */

                track->audio_vbr = 1;

            }else{

                track->sample_size = (av_get_bits_per_sample(st->codecpar->codec_id) >> 3) * st->codecpar->channels;

            }

            if (st->codecpar->codec_id == AV_CODEC_ID_ILBC ||

                st->codecpar->codec_id == AV_CODEC_ID_ADPCM_IMA_QT) {

                track->audio_vbr = 1;

            }

            if (track->mode != MODE_MOV &&

                track->par->codec_id == AV_CODEC_ID_MP3 && track->timescale < 16000) {

                if (s->strict_std_compliance >= FF_COMPLIANCE_NORMAL) {

                    av_log(s, AV_LOG_ERROR, "track %d: muxing mp3 at %dhz is not standard, to mux anyway set strict to -1\n",

                        i, track->par->sample_rate);

                    return AVERROR(EINVAL);

                } else {

                    av_log(s, AV_LOG_WARNING, "track %d: muxing mp3 at %dhz is not standard in MP4\n",

                           i, track->par->sample_rate);

                }

            }

            if (track->par->codec_id == AV_CODEC_ID_FLAC ||

                track->par->codec_id == AV_CODEC_ID_OPUS) {

                if (track->mode != MODE_MP4) {

                    av_log(s, AV_LOG_ERROR, "%s only supported in MP4.\n", avcodec_get_name(track->par->codec_id));

                    return AVERROR(EINVAL);

                }

                if (s->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {

                    av_log(s, AV_LOG_ERROR,

                           "%s in MP4 support is experimental, add "

                           "'-strict %d' if you want to use it.\n",

                           avcodec_get_name(track->par->codec_id), FF_COMPLIANCE_EXPERIMENTAL);

                    return AVERROR_EXPERIMENTAL;

                }

            }

        } else if (st->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) {

            track->timescale = st->time_base.den;

        } else if (st->codecpar->codec_type == AVMEDIA_TYPE_DATA) {

            track->timescale = st->time_base.den;

        } else {

            track->timescale = MOV_TIMESCALE;

        }

        if (!track->height)

            track->height = st->codecpar->height;

        /* The ism specific timescale isn't mandatory, but is assumed by

         * some tools, such as mp4split. */

        if (mov->mode == MODE_ISM)

            track->timescale = 10000000;



        avpriv_set_pts_info(st, 64, 1, track->timescale);



        if (mov->encryption_scheme == MOV_ENC_CENC_AES_CTR) {

            ret = ff_mov_cenc_init(&track->cenc, mov->encryption_key,

                track->par->codec_id == AV_CODEC_ID_H264, s->flags & AVFMT_FLAG_BITEXACT);

            if (ret)

                return ret;

        }

    }



    enable_tracks(s);

    return 0;

}
