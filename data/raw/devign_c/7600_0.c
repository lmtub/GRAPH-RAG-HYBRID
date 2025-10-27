static int yop_probe(AVProbeData *probe_packet)

{

    if (AV_RB16(probe_packet->buf) == AV_RB16("YO")  &&

        probe_packet->buf[6]                         &&

        probe_packet->buf[7]                         &&

        !(probe_packet->buf[8] & 1)                  &&

        !(probe_packet->buf[10] & 1))

        return AVPROBE_SCORE_MAX * 3 / 4;



    return 0;

}
