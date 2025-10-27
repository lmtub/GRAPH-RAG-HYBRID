static uint64_t arm_thiswdog_read(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    arm_mptimer_state *s = (arm_mptimer_state *)opaque;

    int id = get_current_cpu(s);

    return timerblock_read(&s->timerblock[id * 2 + 1], addr, size);

}
