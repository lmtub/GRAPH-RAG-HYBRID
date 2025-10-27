static int mv_read_header(AVFormatContext *avctx)

{

    MvContext *mv = avctx->priv_data;

    AVIOContext *pb = avctx->pb;

    AVStream *ast = NULL, *vst = NULL; //initialization to suppress warning

    int version, i;



    avio_skip(pb, 4);



    version = avio_rb16(pb);

    if (version == 2) {

        uint64_t timestamp;

        int v;

        avio_skip(pb, 22);



        /* allocate audio track first to prevent unnecessary seeking

           (audio packet always precede video packet for a given frame) */

        ast = avformat_new_stream(avctx, NULL);

        if (!ast)

            return AVERROR(ENOMEM);



        vst = avformat_new_stream(avctx, NULL);

        if (!vst)

            return AVERROR(ENOMEM);

        vst->codec->codec_type = AVMEDIA_TYPE_VIDEO;

        vst->time_base = (AVRational){1, 15};

        vst->nb_frames = avio_rb32(pb);

        v = avio_rb32(pb);

        switch (v) {

        case 1:

            vst->codec->codec_id = AV_CODEC_ID_MVC1;

            break;

        case 2:

            vst->codec->pix_fmt  = AV_PIX_FMT_ARGB;

            vst->codec->codec_id = AV_CODEC_ID_RAWVIDEO;

            break;

        default:

            av_log_ask_for_sample(avctx, "unknown video compression %i\n", v);

            break;

        }

        vst->codec->codec_tag = 0;

        vst->codec->width     = avio_rb32(pb);

        vst->codec->height    = avio_rb32(pb);

        avio_skip(pb, 12);



        ast->codec->codec_type     = AVMEDIA_TYPE_AUDIO;

        ast->nb_frames             = vst->nb_frames;

        ast->codec->sample_rate    = avio_rb32(pb);

        avpriv_set_pts_info(ast, 33, 1, ast->codec->sample_rate);

        ast->codec->channels       = avio_rb32(pb);

        ast->codec->channel_layout = (ast->codec->channels == 1) ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;

        v = avio_rb32(pb);

        if (v == AUDIO_FORMAT_SIGNED) {

            ast->codec->codec_id = AV_CODEC_ID_PCM_S16BE;

        } else {

            av_log_ask_for_sample(avctx, "unknown audio compression (format %i)\n", v);

        }



        avio_skip(pb, 12);

        var_read_metadata(avctx, "title", 0x80);

        var_read_metadata(avctx, "comment", 0x100);

        avio_skip(pb, 0x80);



        timestamp = 0;

        for (i = 0; i < vst->nb_frames; i++) {

            uint32_t pos   = avio_rb32(pb);

            uint32_t asize = avio_rb32(pb);

            uint32_t vsize = avio_rb32(pb);

            avio_skip(pb, 8);

            av_add_index_entry(ast, pos,         timestamp, asize, 0, AVINDEX_KEYFRAME);

            av_add_index_entry(vst, pos + asize, i,         vsize, 0, AVINDEX_KEYFRAME);

            timestamp += asize / (ast->codec->channels * 2);

        }

    } else if (!version && avio_rb16(pb) == 3) {

        avio_skip(pb, 4);



        read_table(avctx, NULL, parse_global_var);



        if (mv->nb_audio_tracks > 1) {

            av_log_ask_for_sample(avctx, "multiple audio streams\n");

            return AVERROR_PATCHWELCOME;

        } else if (mv->nb_audio_tracks) {

            ast = avformat_new_stream(avctx, NULL);

            if (!ast)

                return AVERROR(ENOMEM);

            ast->codec->codec_type = AVMEDIA_TYPE_AUDIO;

            /* temporarily store compression value in codec_tag; format value in codec_id */

            read_table(avctx, ast, parse_audio_var);

            if (ast->codec->codec_tag == 100 && ast->codec->codec_id == AUDIO_FORMAT_SIGNED && ast->codec->bits_per_coded_sample == 16) {

                ast->codec->codec_id = AV_CODEC_ID_PCM_S16BE;

            } else {

                av_log_ask_for_sample(avctx, "unknown audio compression %i (format %i, width %i)\n",

                    ast->codec->codec_tag, ast->codec->codec_id, ast->codec->bits_per_coded_sample);

                ast->codec->codec_id = AV_CODEC_ID_NONE;

            }

            ast->codec->codec_tag = 0;

        }



        if (mv->nb_video_tracks > 1) {

            av_log_ask_for_sample(avctx, "multiple video streams\n");

            return AVERROR_PATCHWELCOME;

        } else if (mv->nb_video_tracks) {

            vst = avformat_new_stream(avctx, NULL);

            if (!vst)

                return AVERROR(ENOMEM);

            vst->codec->codec_type = AVMEDIA_TYPE_VIDEO;

            read_table(avctx, vst, parse_video_var);

        }



        if (mv->nb_audio_tracks)

            read_index(pb, ast);



        if (mv->nb_video_tracks)

            read_index(pb, vst);

    } else {

        av_log_ask_for_sample(avctx, "unknown version %i\n", version);

        return AVERROR_PATCHWELCOME;

    }



    return 0;

}
