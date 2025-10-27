static int get_codec_data(ByteIOContext *pb, AVStream *vst,

                          AVStream *ast, int myth) {

    frametype_t frametype;

    if (!vst && !myth)

        return 1; // no codec data needed

    while (!url_feof(pb)) {

        int size, subtype;

        frametype = get_byte(pb);

        switch (frametype) {

            case NUV_EXTRADATA:

                subtype = get_byte(pb);

                url_fskip(pb, 6);

                size = PKTSIZE(get_le32(pb));

                if (subtype == 'R') {

                    vst->codec->extradata_size = size;

                    vst->codec->extradata = av_malloc(size);

                    get_buffer(pb, vst->codec->extradata, size);

                    size = 0;

                    if (!myth)

                        return 1;

                }

                break;

            case NUV_MYTHEXT:

                url_fskip(pb, 7);

                size = PKTSIZE(get_le32(pb));

                if (size != 128 * 4)

                    break;

                get_le32(pb); // version

                if (vst) {

                    vst->codec->codec_tag = get_le32(pb);

                    vst->codec->codec_id =

                        codec_get_id(codec_bmp_tags, vst->codec->codec_tag);

                } else

                    url_fskip(pb, 4);



                if (ast) {

                    ast->codec->codec_tag = get_le32(pb);

                    ast->codec->sample_rate = get_le32(pb);

                    ast->codec->bits_per_sample = get_le32(pb);

                    ast->codec->channels = get_le32(pb);

                    ast->codec->codec_id =

                        wav_codec_get_id(ast->codec->codec_tag,

                                         ast->codec->bits_per_sample);

                } else

                    url_fskip(pb, 4 * 4);



                size -= 6 * 4;

                url_fskip(pb, size);

                return 1;

            case NUV_SEEKP:

                size = 11;

                break;

            default:

                url_fskip(pb, 7);

                size = PKTSIZE(get_le32(pb));

                break;

        }

        url_fskip(pb, size);

    }

    return 0;

}
