static void rtas_ibm_get_system_parameter(PowerPCCPU *cpu,

                                          sPAPRMachineState *spapr,

                                          uint32_t token, uint32_t nargs,

                                          target_ulong args,

                                          uint32_t nret, target_ulong rets)

{

    target_ulong parameter = rtas_ld(args, 0);

    target_ulong buffer = rtas_ld(args, 1);

    target_ulong length = rtas_ld(args, 2);

    target_ulong ret;



    switch (parameter) {

    case RTAS_SYSPARM_SPLPAR_CHARACTERISTICS: {

        char *param_val = g_strdup_printf("MaxEntCap=%d,"

                                          "DesMem=%llu,"

                                          "DesProcs=%d,"

                                          "MaxPlatProcs=%d",

                                          max_cpus,

                                          current_machine->ram_size / M_BYTE,

                                          smp_cpus,

                                          max_cpus);

        ret = sysparm_st(buffer, length, param_val, strlen(param_val) + 1);

        g_free(param_val);

        break;

    }

    case RTAS_SYSPARM_DIAGNOSTICS_RUN_MODE: {

        uint8_t param_val = DIAGNOSTICS_RUN_MODE_DISABLED;



        ret = sysparm_st(buffer, length, &param_val, sizeof(param_val));

        break;

    }

    case RTAS_SYSPARM_UUID:

        ret = sysparm_st(buffer, length, qemu_uuid, (qemu_uuid_set ? 16 : 0));

        break;

    default:

        ret = RTAS_OUT_NOT_SUPPORTED;

    }



    rtas_st(rets, 0, ret);

}
