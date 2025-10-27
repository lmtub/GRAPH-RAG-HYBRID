static int mov_write_avcc_tag(ByteIOContext *pb, MOVTrack *track)

{

    offset_t pos = url_ftell(pb);



    put_be32(pb, 0);

    put_tag(pb, "avcC");

    if (track->vosLen > 6) {

        /* check for h264 start code */

        if (AV_RB32(track->vosData) == 0x00000001) {

            uint8_t *buf, *end;

            uint32_t sps_size=0, pps_size=0;

            uint8_t *sps=0, *pps=0;



            avc_parse_nal_units(&track->vosData, &track->vosLen);

            buf = track->vosData;

            end = track->vosData + track->vosLen;



            /* look for sps and pps */

            while (buf < end) {

                unsigned int size;

                uint8_t nal_type;

                size = AV_RB32(buf);

                nal_type = buf[4] & 0x1f;

                if (nal_type == 7) { /* SPS */

                    sps = buf + 4;

                    sps_size = size;

                } else if (nal_type == 8) { /* PPS */

                    pps = buf + 4;

                    pps_size = size;

                }

                buf += size + 4;

            }

            assert(sps);

            assert(pps);



            put_byte(pb, 1); /* version */

            put_byte(pb, sps[1]); /* profile */

            put_byte(pb, sps[2]); /* profile compat */

            put_byte(pb, sps[3]); /* level */

            put_byte(pb, 0xff); /* 6 bits reserved (111111) + 2 bits nal size length - 1 (11) */

            put_byte(pb, 0xe1); /* 3 bits reserved (111) + 5 bits number of sps (00001) */



            put_be16(pb, sps_size);

            put_buffer(pb, sps, sps_size);

            put_byte(pb, 1); /* number of pps */

            put_be16(pb, pps_size);

            put_buffer(pb, pps, pps_size);

        } else {

            put_buffer(pb, track->vosData, track->vosLen);

        }

    }

    return updateSize(pb, pos);

}
