static int mov_write_mvhd_tag(ByteIOContext *pb, MOVContext *mov)

{

    int maxTrackID = 1, i;

    int64_t maxTrackLenTemp, maxTrackLen = 0;

    int version;



    for (i=0; i<MAX_STREAMS; i++) {

        if(mov->tracks[i].entry > 0) {

            maxTrackLenTemp = av_rescale_rnd(mov->tracks[i].trackDuration, globalTimescale, mov->tracks[i].timescale, AV_ROUND_UP);

            if(maxTrackLen < maxTrackLenTemp)

                maxTrackLen = maxTrackLenTemp;

            if(maxTrackID < mov->tracks[i].trackID)

                maxTrackID = mov->tracks[i].trackID;

        }

    }



    version = maxTrackLen < UINT32_MAX ? 0 : 1;

    (version == 1) ? put_be32(pb, 120) : put_be32(pb, 108); /* size */

    put_tag(pb, "mvhd");

    put_byte(pb, version);

    put_be24(pb, 0); /* flags */

    if (version == 1) {

        put_be64(pb, mov->time);

        put_be64(pb, mov->time);

    } else {

        put_be32(pb, mov->time); /* creation time */

        put_be32(pb, mov->time); /* modification time */

    }

    put_be32(pb, mov->timescale); /* timescale */

    (version == 1) ? put_be64(pb, maxTrackLen) : put_be32(pb, maxTrackLen); /* duration of longest track */



    put_be32(pb, 0x00010000); /* reserved (preferred rate) 1.0 = normal */

    put_be16(pb, 0x0100); /* reserved (preferred volume) 1.0 = normal */

    put_be16(pb, 0); /* reserved */

    put_be32(pb, 0); /* reserved */

    put_be32(pb, 0); /* reserved */



    /* Matrix structure */

    put_be32(pb, 0x00010000); /* reserved */

    put_be32(pb, 0x0); /* reserved */

    put_be32(pb, 0x0); /* reserved */

    put_be32(pb, 0x0); /* reserved */

    put_be32(pb, 0x00010000); /* reserved */

    put_be32(pb, 0x0); /* reserved */

    put_be32(pb, 0x0); /* reserved */

    put_be32(pb, 0x0); /* reserved */

    put_be32(pb, 0x40000000); /* reserved */



    put_be32(pb, 0); /* reserved (preview time) */

    put_be32(pb, 0); /* reserved (preview duration) */

    put_be32(pb, 0); /* reserved (poster time) */

    put_be32(pb, 0); /* reserved (selection time) */

    put_be32(pb, 0); /* reserved (selection duration) */

    put_be32(pb, 0); /* reserved (current time) */

    put_be32(pb, maxTrackID+1); /* Next track id */

    return 0x6c;

}
