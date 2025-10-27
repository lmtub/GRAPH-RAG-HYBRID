void slirp_output(const uint8_t *pkt, int pkt_len)

{

#ifdef DEBUG_SLIRP

    printf("slirp output:\n");

    hex_dump(stdout, pkt, pkt_len);

#endif

    if (!slirp_vc)

        return;

    qemu_send_packet(slirp_vc, pkt, pkt_len);

}
