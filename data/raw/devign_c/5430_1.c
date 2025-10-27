static int mxf_read_track(MXFTrack *track, ByteIOContext *pb, int tag)

{

    switch(tag) {

    case 0x4801:

        track->track_id = get_be32(pb);

        break;

    case 0x4804:

        get_buffer(pb, track->track_number, 4);

        break;

    case 0x4B01:

        track->edit_rate.den = get_be32(pb);

        track->edit_rate.num = get_be32(pb);

        break;

    case 0x4803:

        get_buffer(pb, track->sequence_ref, 16);

        break;

    }

    return 0;

}
