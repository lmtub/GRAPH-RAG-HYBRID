static int ape_tag_read_field(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    uint8_t key[1024], *value;

    uint32_t size, flags;

    int i, c;



    size = avio_rl32(pb);  /* field size */

    flags = avio_rl32(pb); /* field flags */

    for (i = 0; i < sizeof(key) - 1; i++) {

        c = avio_r8(pb);

        if (c < 0x20 || c > 0x7E)

            break;

        else

            key[i] = c;

    }

    key[i] = 0;

    if (c != 0) {

        av_log(s, AV_LOG_WARNING, "Invalid APE tag key '%s'.\n", key);

        return -1;

    }

    if (size > INT32_MAX - FF_INPUT_BUFFER_PADDING_SIZE) {

        av_log(s, AV_LOG_ERROR, "APE tag size too large.\n");

        return AVERROR_INVALIDDATA;

    }

    if (flags & APE_TAG_FLAG_IS_BINARY) {

        uint8_t filename[1024];

        enum AVCodecID id;

        AVStream *st = avformat_new_stream(s, NULL);

        if (!st)

            return AVERROR(ENOMEM);



        size -= avio_get_str(pb, size, filename, sizeof(filename));

        if (size <= 0) {

            av_log(s, AV_LOG_WARNING, "Skipping binary tag '%s'.\n", key);

            return 0;

        }



        av_dict_set(&st->metadata, key, filename, 0);



        if ((id = ff_guess_image2_codec(filename)) != AV_CODEC_ID_NONE) {

            AVPacket pkt;

            int ret;



            ret = av_get_packet(s->pb, &pkt, size);

            if (ret < 0) {

                av_log(s, AV_LOG_ERROR, "Error reading cover art.\n");

                return ret;

            }



            st->disposition      |= AV_DISPOSITION_ATTACHED_PIC;

            st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

            st->codec->codec_id   = id;



            st->attached_pic              = pkt;

            st->attached_pic.stream_index = st->index;

            st->attached_pic.flags       |= AV_PKT_FLAG_KEY;

        } else {

            if (ff_get_extradata(st->codec, s->pb, size) < 0)

                return AVERROR(ENOMEM);

            st->codec->codec_type = AVMEDIA_TYPE_ATTACHMENT;

        }

    } else {

        value = av_malloc(size+1);

        if (!value)

            return AVERROR(ENOMEM);

        c = avio_read(pb, value, size);

        if (c < 0) {

            av_free(value);

            return c;

        }

        value[c] = 0;

        av_dict_set(&s->metadata, key, value, AV_DICT_DONT_STRDUP_VAL);

    }

    return 0;

}
