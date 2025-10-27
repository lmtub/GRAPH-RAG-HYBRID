static int mxf_read_material_package(void *arg, AVIOContext *pb, int tag, int size, UID uid)

{

    MXFPackage *package = arg;

    switch(tag) {

    case 0x4403:

        package->tracks_count = avio_rb32(pb);

        if (package->tracks_count >= UINT_MAX / sizeof(UID))

            return -1;

        package->tracks_refs = av_malloc(package->tracks_count * sizeof(UID));

        if (!package->tracks_refs)

            return -1;

        avio_skip(pb, 4); /* useless size of objects, always 16 according to specs */

        avio_read(pb, (uint8_t *)package->tracks_refs, package->tracks_count * sizeof(UID));

        break;

    }

    return 0;

}
