static void chassis_control(IPMIBmcSim *ibs,

                            uint8_t *cmd, unsigned int cmd_len,

                            uint8_t *rsp, unsigned int *rsp_len,

                            unsigned int max_rsp_len)

{

    IPMIInterface *s = ibs->parent.intf;

    IPMIInterfaceClass *k = IPMI_INTERFACE_GET_CLASS(s);



    IPMI_CHECK_CMD_LEN(3);

    switch (cmd[2] & 0xf) {

    case 0: /* power down */

        rsp[2] = k->do_hw_op(s, IPMI_POWEROFF_CHASSIS, 0);

        break;

    case 1: /* power up */

        rsp[2] = k->do_hw_op(s, IPMI_POWERON_CHASSIS, 0);

        break;

    case 2: /* power cycle */

        rsp[2] = k->do_hw_op(s, IPMI_POWERCYCLE_CHASSIS, 0);

        break;

    case 3: /* hard reset */

        rsp[2] = k->do_hw_op(s, IPMI_RESET_CHASSIS, 0);

        break;

    case 4: /* pulse diagnostic interrupt */

        rsp[2] = k->do_hw_op(s, IPMI_PULSE_DIAG_IRQ, 0);

        break;

    case 5: /* soft shutdown via ACPI by overtemp emulation */

        rsp[2] = k->do_hw_op(s,

                             IPMI_SHUTDOWN_VIA_ACPI_OVERTEMP, 0);

        break;

    default:

        rsp[2] = IPMI_CC_INVALID_DATA_FIELD;

        return;

    }

}
