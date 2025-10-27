static void pvpanic_fw_cfg(ISADevice *dev, FWCfgState *fw_cfg)

{

    PVPanicState *s = ISA_PVPANIC_DEVICE(dev);



    fw_cfg_add_file(fw_cfg, "etc/pvpanic-port",

                    g_memdup(&s->ioport, sizeof(s->ioport)),

                    sizeof(s->ioport));

}
