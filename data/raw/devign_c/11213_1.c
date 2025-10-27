static void get_sel_entry(IPMIBmcSim *ibs,

                          uint8_t *cmd, unsigned int cmd_len,

                          uint8_t *rsp, unsigned int *rsp_len,

                          unsigned int max_rsp_len)

{

    unsigned int val;



    IPMI_CHECK_CMD_LEN(8);

    if (cmd[6]) {

        IPMI_CHECK_RESERVATION(2, ibs->sel.reservation);

    }

    if (ibs->sel.next_free == 0) {

        rsp[2] = IPMI_CC_REQ_ENTRY_NOT_PRESENT;

        return;

    }

    if (cmd[6] > 15) {

        rsp[2] = IPMI_CC_INVALID_DATA_FIELD;

        return;

    }

    if (cmd[7] == 0xff) {

        cmd[7] = 16;

    } else if ((cmd[7] + cmd[6]) > 16) {

        rsp[2] = IPMI_CC_INVALID_DATA_FIELD;

        return;

    } else {

        cmd[7] += cmd[6];

    }



    val = cmd[4] | (cmd[5] << 8);

    if (val == 0xffff) {

        val = ibs->sel.next_free - 1;

    } else if (val >= ibs->sel.next_free) {

        rsp[2] = IPMI_CC_REQ_ENTRY_NOT_PRESENT;

        return;

    }

    if ((val + 1) == ibs->sel.next_free) {

        IPMI_ADD_RSP_DATA(0xff);

        IPMI_ADD_RSP_DATA(0xff);

    } else {

        IPMI_ADD_RSP_DATA((val + 1) & 0xff);

        IPMI_ADD_RSP_DATA(((val + 1) >> 8) & 0xff);

    }

    for (; cmd[6] < cmd[7]; cmd[6]++) {

        IPMI_ADD_RSP_DATA(ibs->sel.sel[val][cmd[6]]);

    }

}
