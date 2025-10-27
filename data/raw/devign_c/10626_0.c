static int mov_read_sv3d(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    MOVStreamContext *sc;

    int size;

    int32_t yaw, pitch, roll;

    size_t l = 0, t = 0, r = 0, b = 0;

    size_t padding = 0;

    uint32_t tag;

    enum AVSphericalProjection projection;



    if (c->fc->nb_streams < 1)

        return 0;



    st = c->fc->streams[c->fc->nb_streams - 1];

    sc = st->priv_data;



    if (atom.size < 8) {

        av_log(c->fc, AV_LOG_ERROR, "Empty spherical video box\n");

        return AVERROR_INVALIDDATA;

    }



    size = avio_rb32(pb);

    if (size <= 12 || size > atom.size)

        return AVERROR_INVALIDDATA;



    tag = avio_rl32(pb);

    if (tag != MKTAG('s','v','h','d')) {

        av_log(c->fc, AV_LOG_ERROR, "Missing spherical video header\n");

        return 0;

    }

    avio_skip(pb, 4); /*  version + flags */

    avio_skip(pb, size - 12); /* metadata_source */



    size = avio_rb32(pb);

    if (size > atom.size)

        return AVERROR_INVALIDDATA;



    tag = avio_rl32(pb);

    if (tag != MKTAG('p','r','o','j')) {

        av_log(c->fc, AV_LOG_ERROR, "Missing projection box\n");

        return 0;

    }



    size = avio_rb32(pb);

    if (size > atom.size)

        return AVERROR_INVALIDDATA;



    tag = avio_rl32(pb);

    if (tag != MKTAG('p','r','h','d')) {

        av_log(c->fc, AV_LOG_ERROR, "Missing projection header box\n");

        return 0;

    }

    avio_skip(pb, 4); /*  version + flags */



    /* 16.16 fixed point */

    yaw   = avio_rb32(pb);

    pitch = avio_rb32(pb);

    roll  = avio_rb32(pb);



    size = avio_rb32(pb);

    if (size > atom.size)

        return AVERROR_INVALIDDATA;



    tag = avio_rl32(pb);

    avio_skip(pb, 4); /*  version + flags */

    switch (tag) {

    case MKTAG('c','b','m','p'):

        projection = AV_SPHERICAL_CUBEMAP;

        padding = avio_rb32(pb);

        break;

    case MKTAG('e','q','u','i'):

        t = avio_rb32(pb);

        b = avio_rb32(pb);

        l = avio_rb32(pb);

        r = avio_rb32(pb);



        if (b >= UINT_MAX - t || r >= UINT_MAX - l) {

            av_log(c->fc, AV_LOG_ERROR,

                   "Invalid bounding rectangle coordinates %"SIZE_SPECIFIER","

                   "%"SIZE_SPECIFIER",%"SIZE_SPECIFIER",%"SIZE_SPECIFIER"\n",

                   l, t, r, b);

            return AVERROR_INVALIDDATA;

        }



        if (l || t || r || b)

            projection = AV_SPHERICAL_EQUIRECTANGULAR_TILE;

        else

            projection = AV_SPHERICAL_EQUIRECTANGULAR;

        break;

    default:

        av_log(c->fc, AV_LOG_ERROR, "Unknown projection type\n");

        return 0;

    }



    sc->spherical = av_spherical_alloc(&sc->spherical_size);

    if (!sc->spherical)

        return AVERROR(ENOMEM);



    sc->spherical->projection = projection;



    sc->spherical->yaw   = yaw;

    sc->spherical->pitch = pitch;

    sc->spherical->roll  = roll;



    sc->spherical->padding = padding;



    sc->spherical->bound_left   = l;

    sc->spherical->bound_top    = t;

    sc->spherical->bound_right  = r;

    sc->spherical->bound_bottom = b;



    return 0;

}
