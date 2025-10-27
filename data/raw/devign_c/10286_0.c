static void armv7m_nvic_reset(DeviceState *dev)

{

    nvic_state *s = NVIC(dev);

    NVICClass *nc = NVIC_GET_CLASS(s);

    nc->parent_reset(dev);

    /* Common GIC reset resets to disabled; the NVIC doesn't have

     * per-CPU interfaces so mark our non-existent CPU interface

     * as enabled by default, and with a priority mask which allows

     * all interrupts through.

     */

    s->gic.cpu_enabled[0] = true;

    s->gic.priority_mask[0] = 0x100;

    /* The NVIC as a whole is always enabled. */

    s->gic.enabled = true;

    systick_reset(s);

}
