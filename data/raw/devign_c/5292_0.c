static int handle_sigp_single_dst(S390CPU *dst_cpu, uint8_t order,

                                  uint64_t param, uint64_t *status_reg)

{

    SigpInfo si = {

        .param = param,

        .status_reg = status_reg,

    };



    /* cpu available? */

    if (dst_cpu == NULL) {

        return SIGP_CC_NOT_OPERATIONAL;

    }



    /* only resets can break pending orders */

    if (dst_cpu->env.sigp_order != 0 &&

        order != SIGP_CPU_RESET &&

        order != SIGP_INITIAL_CPU_RESET) {

        return SIGP_CC_BUSY;

    }



    switch (order) {

    case SIGP_START:

        run_on_cpu(CPU(dst_cpu), sigp_start, RUN_ON_CPU_HOST_PTR(&si));

        break;

    case SIGP_STOP:

        run_on_cpu(CPU(dst_cpu), sigp_stop, RUN_ON_CPU_HOST_PTR(&si));

        break;

    case SIGP_RESTART:

        run_on_cpu(CPU(dst_cpu), sigp_restart, RUN_ON_CPU_HOST_PTR(&si));

        break;

    case SIGP_STOP_STORE_STATUS:

        run_on_cpu(CPU(dst_cpu), sigp_stop_and_store_status, RUN_ON_CPU_HOST_PTR(&si));

        break;

    case SIGP_STORE_STATUS_ADDR:

        run_on_cpu(CPU(dst_cpu), sigp_store_status_at_address, RUN_ON_CPU_HOST_PTR(&si));

        break;

    case SIGP_STORE_ADTL_STATUS:

        run_on_cpu(CPU(dst_cpu), sigp_store_adtl_status, RUN_ON_CPU_HOST_PTR(&si));

        break;

    case SIGP_SET_PREFIX:

        run_on_cpu(CPU(dst_cpu), sigp_set_prefix, RUN_ON_CPU_HOST_PTR(&si));

        break;

    case SIGP_INITIAL_CPU_RESET:

        run_on_cpu(CPU(dst_cpu), sigp_initial_cpu_reset, RUN_ON_CPU_HOST_PTR(&si));

        break;

    case SIGP_CPU_RESET:

        run_on_cpu(CPU(dst_cpu), sigp_cpu_reset, RUN_ON_CPU_HOST_PTR(&si));

        break;

    default:

        set_sigp_status(&si, SIGP_STAT_INVALID_ORDER);

    }



    return si.cc;

}
