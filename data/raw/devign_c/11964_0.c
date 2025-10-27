static uint64_t get_guest_rtc_ns(RTCState *s)

{

    uint64_t guest_rtc;

    uint64_t guest_clock = qemu_clock_get_ns(rtc_clock);



    guest_rtc = s->base_rtc * NANOSECONDS_PER_SECOND +

        guest_clock - s->last_update + s->offset;

    return guest_rtc;

}
