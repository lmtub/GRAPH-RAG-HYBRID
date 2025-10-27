int arm_reset_cpu(uint64_t cpuid)

{

    CPUState *target_cpu_state;

    ARMCPU *target_cpu;



    DPRINTF("cpu %" PRId64 "\n", cpuid);



    /* change to the cpu we are resetting */

    target_cpu_state = arm_get_cpu_by_id(cpuid);

    if (!target_cpu_state) {

        return QEMU_ARM_POWERCTL_INVALID_PARAM;

    }

    target_cpu = ARM_CPU(target_cpu_state);

    if (target_cpu->powered_off) {

        qemu_log_mask(LOG_GUEST_ERROR,

                      "[ARM]%s: CPU %" PRId64 " is off\n",

                      __func__, cpuid);

        return QEMU_ARM_POWERCTL_IS_OFF;

    }



    /* Reset the cpu */

    cpu_reset(target_cpu_state);



    return QEMU_ARM_POWERCTL_RET_SUCCESS;

}
