static void fw_cfg_initfn(Object *obj)

{

    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    FWCfgState *s = FW_CFG(obj);



    memory_region_init_io(&s->ctl_iomem, OBJECT(s), &fw_cfg_ctl_mem_ops, s,

                          "fwcfg.ctl", FW_CFG_SIZE);

    sysbus_init_mmio(sbd, &s->ctl_iomem);

    memory_region_init_io(&s->data_iomem, OBJECT(s), &fw_cfg_data_mem_ops, s,

                          "fwcfg.data", FW_CFG_DATA_SIZE);

    sysbus_init_mmio(sbd, &s->data_iomem);

    /* In case ctl and data overlap: */

    memory_region_init_io(&s->comb_iomem, OBJECT(s), &fw_cfg_comb_mem_ops, s,

                          "fwcfg", FW_CFG_SIZE);

}
