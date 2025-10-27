static int read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    BinkDemuxContext *bink = s->priv_data;

    AVIOContext *pb = s->pb;

    uint32_t fps_num, fps_den;

    AVStream *vst, *ast;

    unsigned int i;

    uint32_t pos, next_pos;

    uint16_t flags;

    int keyframe;



    vst = av_new_stream(s, 0);

    if (!vst)

        return AVERROR(ENOMEM);



    vst->codec->codec_tag = avio_rl32(pb);



    bink->file_size = avio_rl32(pb) + 8;

    vst->duration   = avio_rl32(pb);



    if (vst->duration > 1000000) {

        av_log(s, AV_LOG_ERROR, "invalid header: more than 1000000 frames\n");

        return AVERROR(EIO);

    }



    if (avio_rl32(pb) > bink->file_size) {

        av_log(s, AV_LOG_ERROR,

               "invalid header: largest frame size greater than file size\n");

        return AVERROR(EIO);

    }



    avio_skip(pb, 4);



    vst->codec->width  = avio_rl32(pb);

    vst->codec->height = avio_rl32(pb);



    fps_num = avio_rl32(pb);

    fps_den = avio_rl32(pb);

    if (fps_num == 0 || fps_den == 0) {

        av_log(s, AV_LOG_ERROR, "invalid header: invalid fps (%d/%d)\n", fps_num, fps_den);

        return AVERROR(EIO);

    }

    av_set_pts_info(vst, 64, fps_den, fps_num);



    vst->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    vst->codec->codec_id   = CODEC_ID_BINKVIDEO;

    vst->codec->extradata  = av_mallocz(4 + FF_INPUT_BUFFER_PADDING_SIZE);

    vst->codec->extradata_size = 4;

    avio_read(pb, vst->codec->extradata, 4);



    bink->num_audio_tracks = avio_rl32(pb);



    if (bink->num_audio_tracks > BINK_MAX_AUDIO_TRACKS) {

        av_log(s, AV_LOG_ERROR,

               "invalid header: more than "AV_STRINGIFY(BINK_MAX_AUDIO_TRACKS)" audio tracks (%d)\n",

               bink->num_audio_tracks);

        return AVERROR(EIO);

    }



    if (bink->num_audio_tracks) {

        avio_skip(pb, 4 * bink->num_audio_tracks);



        for (i = 0; i < bink->num_audio_tracks; i++) {

            ast = av_new_stream(s, 1);

            if (!ast)

                return AVERROR(ENOMEM);

            ast->codec->codec_type  = AVMEDIA_TYPE_AUDIO;

            ast->codec->codec_tag   = vst->codec->codec_tag;

            ast->codec->sample_rate = avio_rl16(pb);

            av_set_pts_info(ast, 64, 1, ast->codec->sample_rate);

            flags = avio_rl16(pb);

            ast->codec->codec_id = flags & BINK_AUD_USEDCT ?

                                   CODEC_ID_BINKAUDIO_DCT : CODEC_ID_BINKAUDIO_RDFT;

            ast->codec->channels = flags & BINK_AUD_STEREO ? 2 : 1;

        }



        for (i = 0; i < bink->num_audio_tracks; i++)

            s->streams[i + 1]->id = avio_rl32(pb);

    }



    /* frame index table */

    next_pos = avio_rl32(pb);

    for (i = 0; i < vst->duration; i++) {

        pos = next_pos;

        if (i == vst->duration - 1) {

            next_pos = bink->file_size;

            keyframe = 0;

        } else {

            next_pos = avio_rl32(pb);

            keyframe = pos & 1;

        }

        pos &= ~1;

        next_pos &= ~1;



        if (next_pos <= pos) {

            av_log(s, AV_LOG_ERROR, "invalid frame index table\n");

            return AVERROR(EIO);

        }

        av_add_index_entry(vst, pos, i, next_pos - pos, 0,

                           keyframe ? AVINDEX_KEYFRAME : 0);

    }



    avio_skip(pb, 4);



    bink->current_track = -1;

    return 0;

}
