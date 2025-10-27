static int mov_read_uuid(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    MOVStreamContext *sc;

    int64_t ret;

    uint8_t uuid[16];

    static const uint8_t uuid_isml_manifest[] = {

        0xa5, 0xd4, 0x0b, 0x30, 0xe8, 0x14, 0x11, 0xdd,

        0xba, 0x2f, 0x08, 0x00, 0x20, 0x0c, 0x9a, 0x66

    };

    static const uint8_t uuid_xmp[] = {

        0xbe, 0x7a, 0xcf, 0xcb, 0x97, 0xa9, 0x42, 0xe8,

        0x9c, 0x71, 0x99, 0x94, 0x91, 0xe3, 0xaf, 0xac

    };

    static const uint8_t uuid_spherical[] = {

        0xff, 0xcc, 0x82, 0x63, 0xf8, 0x55, 0x4a, 0x93,

        0x88, 0x14, 0x58, 0x7a, 0x02, 0x52, 0x1f, 0xdd,

    };



    if (atom.size < sizeof(uuid) || atom.size == INT64_MAX)

        return AVERROR_INVALIDDATA;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams - 1];

    sc = st->priv_data;



    ret = avio_read(pb, uuid, sizeof(uuid));

    if (ret < 0) {

        return ret;

    } else if (ret != sizeof(uuid)) {

        return AVERROR_INVALIDDATA;

    }

    if (!memcmp(uuid, uuid_isml_manifest, sizeof(uuid))) {

        uint8_t *buffer, *ptr;

        char *endptr;

        size_t len = atom.size - sizeof(uuid);



        if (len < 4) {

            return AVERROR_INVALIDDATA;

        }

        ret = avio_skip(pb, 4); // zeroes

        len -= 4;



        buffer = av_mallocz(len + 1);

        if (!buffer) {

            return AVERROR(ENOMEM);

        }

        ret = avio_read(pb, buffer, len);

        if (ret < 0) {

            av_free(buffer);

            return ret;

        } else if (ret != len) {

            av_free(buffer);

            return AVERROR_INVALIDDATA;

        }



        ptr = buffer;

        while ((ptr = av_stristr(ptr, "systemBitrate=\""))) {

            ptr += sizeof("systemBitrate=\"") - 1;

            c->bitrates_count++;

            c->bitrates = av_realloc_f(c->bitrates, c->bitrates_count, sizeof(*c->bitrates));

            if (!c->bitrates) {

                c->bitrates_count = 0;

                av_free(buffer);

                return AVERROR(ENOMEM);

            }

            errno = 0;

            ret = strtol(ptr, &endptr, 10);

            if (ret < 0 || errno || *endptr != '"') {

                c->bitrates[c->bitrates_count - 1] = 0;

            } else {

                c->bitrates[c->bitrates_count - 1] = ret;

            }

        }



        av_free(buffer);

    } else if (!memcmp(uuid, uuid_xmp, sizeof(uuid))) {

        uint8_t *buffer;

        size_t len = atom.size - sizeof(uuid);

        if (c->export_xmp) {

            buffer = av_mallocz(len + 1);

            if (!buffer) {

                return AVERROR(ENOMEM);

            }

            ret = avio_read(pb, buffer, len);

            if (ret < 0) {

                av_free(buffer);

                return ret;

            } else if (ret != len) {

                av_free(buffer);

                return AVERROR_INVALIDDATA;

            }

            buffer[len] = '\0';

            av_dict_set(&c->fc->metadata, "xmp", buffer, 0);

            av_free(buffer);

        } else {

            // skip all uuid atom, which makes it fast for long uuid-xmp file

            ret = avio_skip(pb, len);

            if (ret < 0)

                return ret;

        }

    } else if (!memcmp(uuid, uuid_spherical, sizeof(uuid))) {

        size_t len = atom.size - sizeof(uuid);

        ret = mov_parse_uuid_spherical(sc, pb, len);

        if (ret < 0)

            return ret;

        if (!sc->spherical)

            av_log(c->fc, AV_LOG_WARNING, "Invalid spherical metadata found\n");    }



    return 0;

}
