static void arm_gic_realize(DeviceState *dev, Error **errp)

{

    /* Device instance realize function for the GIC sysbus device */

    GICv3State *s = ARM_GICV3(dev);

    ARMGICv3Class *agc = ARM_GICV3_GET_CLASS(s);

    Error *local_err = NULL;



    agc->parent_realize(dev, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    gicv3_init_irqs_and_mmio(s, gicv3_set_irq, NULL);

}
