static int mxf_read_content_storage(void *arg, AVIOContext *pb, int tag, int size, UID uid)

{

    MXFContext *mxf = arg;

    switch (tag) {

    case 0x1901:

        mxf->packages_count = avio_rb32(pb);

        if (mxf->packages_count >= UINT_MAX / sizeof(UID))

            return -1;

        mxf->packages_refs = av_malloc(mxf->packages_count * sizeof(UID));

        if (!mxf->packages_refs)

            return -1;

        avio_skip(pb, 4); /* useless size of objects, always 16 according to specs */

        avio_read(pb, (uint8_t *)mxf->packages_refs, mxf->packages_count * sizeof(UID));

        break;

    }

    return 0;

}
