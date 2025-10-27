static CadenceTimerState *cadence_timer_from_addr(void *opaque,

                                        target_phys_addr_t offset)

{

    unsigned int index;

    CadenceTTCState *s = (CadenceTTCState *)opaque;



    index = (offset >> 2) % 3;



    return &s->timer[index];

}
