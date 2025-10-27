static int mxf_read_track(void *arg, AVIOContext *pb, int tag, int size, UID uid)

{

    MXFTrack *track = arg;

    switch(tag) {

    case 0x4801:

        track->track_id = avio_rb32(pb);

        break;

    case 0x4804:

        avio_read(pb, track->track_number, 4);

        break;

    case 0x4B01:

        track->edit_rate.den = avio_rb32(pb);

        track->edit_rate.num = avio_rb32(pb);

        break;

    case 0x4803:

        avio_read(pb, track->sequence_ref, 16);

        break;

    }

    return 0;

}
