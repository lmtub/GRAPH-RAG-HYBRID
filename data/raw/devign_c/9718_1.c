static int vqf_probe(AVProbeData *probe_packet)
{
    if (AV_RL32(probe_packet->buf) != MKTAG('T','W','I','N'))
        return 0;
    if (!memcmp(probe_packet->buf + 4, "97012000", 8))
        return AVPROBE_SCORE_MAX;
    if (!memcmp(probe_packet->buf + 4, "00052200", 8))
        return AVPROBE_SCORE_MAX;
    return AVPROBE_SCORE_EXTENSION;
}