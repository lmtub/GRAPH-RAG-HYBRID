static void sclp_execute(SCCB *sccb, uint64_t code)

{

    S390SCLPDevice *sdev = get_event_facility();



    switch (code & SCLP_CMD_CODE_MASK) {

    case SCLP_CMDW_READ_SCP_INFO:

    case SCLP_CMDW_READ_SCP_INFO_FORCED:

        read_SCP_info(sccb);

        break;

    case SCLP_CMDW_READ_CPU_INFO:

        sclp_read_cpu_info(sccb);

        break;

    default:

        sdev->sclp_command_handler(sdev->ef, sccb, code);

        break;

    }

}
