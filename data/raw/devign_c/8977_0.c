static int rtc_load_td(QEMUFile *f, void *opaque, int version_id)

{

    RTCState *s = opaque;



    if (version_id != 1)

        return -EINVAL;



    s->irq_coalesced = qemu_get_be32(f);

    s->period = qemu_get_be32(f);

    rtc_coalesced_timer_update(s);

    return 0;

}
