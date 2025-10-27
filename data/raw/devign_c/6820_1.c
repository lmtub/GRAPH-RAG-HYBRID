static ssize_t dump_receive(VLANClientState *vc, const uint8_t *buf, size_t size)

{

    DumpState *s = vc->opaque;

    struct pcap_sf_pkthdr hdr;

    int64_t ts;

    int caplen;



    /* Early return in case of previous error. */

    if (s->fd < 0) {

        return size;

    }



    ts = muldiv64(qemu_get_clock(vm_clock), 1000000, get_ticks_per_sec());

    caplen = size > s->pcap_caplen ? s->pcap_caplen : size;



    hdr.ts.tv_sec = ts / 1000000;

    hdr.ts.tv_usec = ts % 1000000;

    hdr.caplen = caplen;

    hdr.len = size;

    if (write(s->fd, &hdr, sizeof(hdr)) != sizeof(hdr) ||

        write(s->fd, buf, caplen) != caplen) {

        qemu_log("-net dump write error - stop dump\n");

        close(s->fd);

        s->fd = -1;

    }



    return size;

}
