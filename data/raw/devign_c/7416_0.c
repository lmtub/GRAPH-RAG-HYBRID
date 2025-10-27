static void gic_thiscpu_write(void *opaque, target_phys_addr_t addr,

                              uint64_t value, unsigned size)

{

    GICState *s = (GICState *)opaque;

    gic_cpu_write(s, gic_get_current_cpu(s), addr, value);

}
