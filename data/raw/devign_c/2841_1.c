static void fsl_imx31_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);



    dc->realize = fsl_imx31_realize;



    dc->desc = "i.MX31 SOC";

}
