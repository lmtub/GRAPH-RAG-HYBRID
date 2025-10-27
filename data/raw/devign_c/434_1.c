static void tpm_tis_initfn(Object *obj)

{

    ISADevice *dev = ISA_DEVICE(obj);

    TPMState *s = TPM(obj);



    memory_region_init_io(&s->mmio, OBJECT(s), &tpm_tis_memory_ops,

                          s, "tpm-tis-mmio",

                          TPM_TIS_NUM_LOCALITIES << TPM_TIS_LOCALITY_SHIFT);

    memory_region_add_subregion(isa_address_space(dev), TPM_TIS_ADDR_BASE,

                                &s->mmio);

}
