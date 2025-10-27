static int get_codec_data(AVIOContext *pb, AVStream *vst,

                          AVStream *ast, int myth)

{

    nuv_frametype frametype;



    if (!vst && !myth)

        return 1; // no codec data needed

    while (!avio_feof(pb)) {

        int size, subtype;



        frametype = avio_r8(pb);

        switch (frametype) {

        case NUV_EXTRADATA:

            subtype = avio_r8(pb);

            avio_skip(pb, 6);

            size = PKTSIZE(avio_rl32(pb));

            if (vst && subtype == 'R') {

                if (vst->codecpar->extradata) {

                    av_freep(&vst->codecpar->extradata);

                    vst->codecpar->extradata_size = 0;

                }

                if (ff_get_extradata(NULL, vst->codecpar, pb, size) < 0)

                    return AVERROR(ENOMEM);

                size = 0;

                if (!myth)

                    return 0;

            }

            break;

        case NUV_MYTHEXT:

            avio_skip(pb, 7);

            size = PKTSIZE(avio_rl32(pb));

            if (size != 128 * 4)

                break;

            avio_rl32(pb); // version

            if (vst) {

                vst->codecpar->codec_tag = avio_rl32(pb);

                vst->codecpar->codec_id =

                    ff_codec_get_id(ff_codec_bmp_tags, vst->codecpar->codec_tag);

                if (vst->codecpar->codec_tag == MKTAG('R', 'J', 'P', 'G'))

                    vst->codecpar->codec_id = AV_CODEC_ID_NUV;

            } else

                avio_skip(pb, 4);



            if (ast) {

                int id;



                ast->codecpar->codec_tag             = avio_rl32(pb);

                ast->codecpar->sample_rate           = avio_rl32(pb);

                ast->codecpar->bits_per_coded_sample = avio_rl32(pb);

                ast->codecpar->channels              = avio_rl32(pb);

                ast->codecpar->channel_layout        = 0;



                id = ff_wav_codec_get_id(ast->codecpar->codec_tag,

                                         ast->codecpar->bits_per_coded_sample);

                if (id == AV_CODEC_ID_NONE) {

                    id = ff_codec_get_id(nuv_audio_tags, ast->codecpar->codec_tag);

                    if (id == AV_CODEC_ID_PCM_S16LE)

                        id = ff_get_pcm_codec_id(ast->codecpar->bits_per_coded_sample,

                                                 0, 0, ~1);

                }

                ast->codecpar->codec_id = id;



                ast->need_parsing = AVSTREAM_PARSE_FULL;

            } else

                avio_skip(pb, 4 * 4);



            size -= 6 * 4;

            avio_skip(pb, size);

            return 0;

        case NUV_SEEKP:

            size = 11;

            break;

        default:

            avio_skip(pb, 7);

            size = PKTSIZE(avio_rl32(pb));

            break;

        }

        avio_skip(pb, size);

    }



    return 0;

}
