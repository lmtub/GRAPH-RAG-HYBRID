static char *spapr_vio_get_dev_name(DeviceState *qdev)

{

    VIOsPAPRDevice *dev = VIO_SPAPR_DEVICE(qdev);

    VIOsPAPRDeviceClass *pc = VIO_SPAPR_DEVICE_GET_CLASS(dev);

    char *name;



    /* Device tree style name device@reg */

    name = g_strdup_printf("%s@%x", pc->dt_name, dev->reg);



    return name;

}
