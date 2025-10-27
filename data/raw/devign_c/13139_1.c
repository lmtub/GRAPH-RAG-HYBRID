static int mov_read_custom(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    int64_t end = avio_tell(pb) + atom.size;

    uint8_t *key = NULL, *val = NULL, *mean = NULL;

    int i;

    AVStream *st;

    MOVStreamContext *sc;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];

    sc = st->priv_data;



    for (i = 0; i < 3; i++) {

        uint8_t **p;

        uint32_t len, tag;

        int ret;



        if (end - avio_tell(pb) <= 12)

            break;



        len = avio_rb32(pb);

        tag = avio_rl32(pb);

        avio_skip(pb, 4); // flags



        if (len < 12 || len - 12 > end - avio_tell(pb))

            break;

        len -= 12;



        if (tag == MKTAG('m', 'e', 'a', 'n'))

            p = &mean;

        else if (tag == MKTAG('n', 'a', 'm', 'e'))

            p = &key;

        else if (tag == MKTAG('d', 'a', 't', 'a') && len > 4) {

            avio_skip(pb, 4);

            len -= 4;

            p = &val;

        } else

            break;



        *p = av_malloc(len + 1);

        if (!*p)

            break;

        ret = ffio_read_size(pb, *p, len);

        if (ret < 0) {

            av_freep(p);

            return ret;

        }

        (*p)[len] = 0;

    }



    if (mean && key && val) {

        if (strcmp(key, "iTunSMPB") == 0) {

            int priming, remainder, samples;

            if(sscanf(val, "%*X %X %X %X", &priming, &remainder, &samples) == 3){

                if(priming>0 && priming<16384)

                    sc->start_pad = priming;

            }

        }

        if (strcmp(key, "cdec") != 0) {

            av_dict_set(&c->fc->metadata, key, val,

                        AV_DICT_DONT_STRDUP_KEY | AV_DICT_DONT_STRDUP_VAL);

            key = val = NULL;

        }

    } else {

        av_log(c->fc, AV_LOG_VERBOSE,

               "Unhandled or malformed custom metadata of size %"PRId64"\n", atom.size);

    }



    avio_seek(pb, end, SEEK_SET);

    av_freep(&key);

    av_freep(&val);

    av_freep(&mean);

    return 0;

}
