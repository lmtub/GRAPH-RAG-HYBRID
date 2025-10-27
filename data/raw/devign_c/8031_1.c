static void arm_gic_common_realize(DeviceState *dev, Error **errp)
{
    GICState *s = ARM_GIC_COMMON(dev);
    int num_irq = s->num_irq;
    if (s->num_cpu > GIC_NCPU) {
        error_setg(errp, "requested %u CPUs exceeds GIC maximum %d",
                   s->num_cpu, GIC_NCPU);
    s->num_irq += GIC_BASE_IRQ;
    if (s->num_irq > GIC_MAXIRQ) {
        error_setg(errp,
                   "requested %u interrupt lines exceeds GIC maximum %d",
                   num_irq, GIC_MAXIRQ);
    /* ITLinesNumber is represented as (N / 32) - 1 (see
     * gic_dist_readb) so this is an implementation imposed
     * restriction, not an architectural one:
     */
    if (s->num_irq < 32 || (s->num_irq % 32)) {
        error_setg(errp,
                   "%d interrupt lines unsupported: not divisible by 32",
                   num_irq);