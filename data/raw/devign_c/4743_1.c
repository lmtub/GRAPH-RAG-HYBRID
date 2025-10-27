static int mov_read_custom_2plus(MOVContext *c, AVIOContext *pb, int size)

{

    int64_t end = avio_tell(pb) + size;

    uint8_t *key = NULL, *val = NULL;

    int i;

    AVStream *st;

    MOVStreamContext *sc;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];

    sc = st->priv_data;



    for (i = 0; i < 2; i++) {

        uint8_t **p;

        uint32_t len, tag;



        if (end - avio_tell(pb) <= 12)

            break;



        len = avio_rb32(pb);

        tag = avio_rl32(pb);

        avio_skip(pb, 4); // flags



        if (len < 12 || len - 12 > end - avio_tell(pb))

            break;

        len -= 12;



        if (tag == MKTAG('n', 'a', 'm', 'e'))

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

        avio_read(pb, *p, len);

        (*p)[len] = 0;

    }



    if (key && val) {

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

    }



    avio_seek(pb, end, SEEK_SET);

    av_freep(&key);

    av_freep(&val);

    return 0;

}
