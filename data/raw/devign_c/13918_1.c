static int mxf_read_source_clip(MXFStructuralComponent *source_clip, ByteIOContext *pb, int tag)

{

    switch(tag) {

    case 0x0202:

        source_clip->duration = get_be64(pb);

        break;

    case 0x1201:

        source_clip->start_position = get_be64(pb);

        break;

    case 0x1101:

        /* UMID, only get last 16 bytes */

        url_fskip(pb, 16);

        get_buffer(pb, source_clip->source_package_uid, 16);

        break;

    case 0x1102:

        source_clip->source_track_id = get_be32(pb);

        break;

    }

    return 0;

}
