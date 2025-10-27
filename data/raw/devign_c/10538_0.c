static void cpu_pre_save(void *opaque)

{

    ARMCPU *cpu = opaque;



    if (!write_cpustate_to_list(cpu)) {

        /* This should never fail. */

        abort();

    }



    cpu->cpreg_vmstate_array_len = cpu->cpreg_array_len;

    memcpy(cpu->cpreg_vmstate_indexes, cpu->cpreg_indexes,

           cpu->cpreg_array_len * sizeof(uint64_t));

    memcpy(cpu->cpreg_vmstate_values, cpu->cpreg_values,

           cpu->cpreg_array_len * sizeof(uint64_t));

}
