static void set_watchdog_timer(IPMIBmcSim *ibs,

                               uint8_t *cmd, unsigned int cmd_len,

                               uint8_t *rsp, unsigned int *rsp_len,

                               unsigned int max_rsp_len)

{

    IPMIInterface *s = ibs->parent.intf;

    IPMIInterfaceClass *k = IPMI_INTERFACE_GET_CLASS(s);

    unsigned int val;



    IPMI_CHECK_CMD_LEN(8);

    val = cmd[2] & 0x7; /* Validate use */

    if (val == 0 || val > 5) {

        rsp[2] = IPMI_CC_INVALID_DATA_FIELD;

        return;

    }

    val = cmd[3] & 0x7; /* Validate action */

    switch (val) {

    case IPMI_BMC_WATCHDOG_ACTION_NONE:

        break;



    case IPMI_BMC_WATCHDOG_ACTION_RESET:

        rsp[2] = k->do_hw_op(s, IPMI_RESET_CHASSIS, 1);

        break;



    case IPMI_BMC_WATCHDOG_ACTION_POWER_DOWN:

        rsp[2] = k->do_hw_op(s, IPMI_POWEROFF_CHASSIS, 1);

        break;



    case IPMI_BMC_WATCHDOG_ACTION_POWER_CYCLE:

        rsp[2] = k->do_hw_op(s, IPMI_POWERCYCLE_CHASSIS, 1);

        break;



    default:

        rsp[2] = IPMI_CC_INVALID_DATA_FIELD;

    }

    if (rsp[2]) {

        rsp[2] = IPMI_CC_INVALID_DATA_FIELD;

        return;

    }



    val = (cmd[3] >> 4) & 0x7; /* Validate preaction */

    switch (val) {

    case IPMI_BMC_WATCHDOG_PRE_MSG_INT:

    case IPMI_BMC_WATCHDOG_PRE_NONE:

        break;



    case IPMI_BMC_WATCHDOG_PRE_NMI:

        if (!k->do_hw_op(s, IPMI_SEND_NMI, 1)) {

            /* NMI not supported. */

            rsp[2] = IPMI_CC_INVALID_DATA_FIELD;

            return;

        }

        break;



    default:

        /* We don't support PRE_SMI */

        rsp[2] = IPMI_CC_INVALID_DATA_FIELD;

        return;

    }



    ibs->watchdog_initialized = 1;

    ibs->watchdog_use = cmd[2] & IPMI_BMC_WATCHDOG_USE_MASK;

    ibs->watchdog_action = cmd[3] & IPMI_BMC_WATCHDOG_ACTION_MASK;

    ibs->watchdog_pretimeout = cmd[4];

    ibs->watchdog_expired &= ~cmd[5];

    ibs->watchdog_timeout = cmd[6] | (((uint16_t) cmd[7]) << 8);

    if (ibs->watchdog_running & IPMI_BMC_WATCHDOG_GET_DONT_STOP(ibs)) {

        do_watchdog_reset(ibs);

    } else {

        ibs->watchdog_running = 0;

    }

}
