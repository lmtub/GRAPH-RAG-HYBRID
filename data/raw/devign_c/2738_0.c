void imx_timerp_create(const target_phys_addr_t addr,

                              qemu_irq irq,

                              DeviceState *ccm)

{

    IMXTimerPState *pp;

    DeviceState *dev;



    dev = sysbus_create_simple("imx_timerp", addr, irq);

    pp = container_of(dev, IMXTimerPState, busdev.qdev);

    pp->ccm = ccm;

}
