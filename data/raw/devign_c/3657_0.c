static target_ulong h_put_term_char(CPUState *env, sPAPREnvironment *spapr,

                                    target_ulong opcode, target_ulong *args)

{

    target_ulong reg = args[0];

    target_ulong len = args[1];

    target_ulong char0_7 = args[2];

    target_ulong char8_15 = args[3];

    VIOsPAPRDevice *sdev = spapr_vio_find_by_reg(spapr->vio_bus, reg);

    uint8_t buf[16];



    if (!sdev) {

        return H_PARAMETER;

    }



    if (len > 16) {

        return H_PARAMETER;

    }



    *((uint64_t *)buf) = cpu_to_be64(char0_7);

    *((uint64_t *)buf + 1) = cpu_to_be64(char8_15);



    vty_putchars(sdev, buf, len);



    return H_SUCCESS;

}
