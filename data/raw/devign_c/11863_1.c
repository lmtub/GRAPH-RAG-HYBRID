static uint64_t hpet_get_ticks(void)

{

    uint64_t ticks;

    ticks = ns_to_ticks(qemu_get_clock(vm_clock) + hpet_statep->hpet_offset);

    return ticks;

}
