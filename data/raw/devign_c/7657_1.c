static void ipmi_sim_handle_command(IPMIBmc *b,

                                    uint8_t *cmd, unsigned int cmd_len,

                                    unsigned int max_cmd_len,

                                    uint8_t msg_id)

{

    IPMIBmcSim *ibs = IPMI_BMC_SIMULATOR(b);

    IPMIInterface *s = ibs->parent.intf;

    IPMIInterfaceClass *k = IPMI_INTERFACE_GET_CLASS(s);

    unsigned int netfn;

    uint8_t rsp[MAX_IPMI_MSG_SIZE];

    unsigned int rsp_len_holder = 0;

    unsigned int *rsp_len = &rsp_len_holder;

    unsigned int max_rsp_len = sizeof(rsp);



    /* Set up the response, set the low bit of NETFN. */

    /* Note that max_rsp_len must be at least 3 */

    if (max_rsp_len < 3) {

        rsp[2] = IPMI_CC_REQUEST_DATA_TRUNCATED;

        goto out;

    }



    IPMI_ADD_RSP_DATA(cmd[0] | 0x04);

    IPMI_ADD_RSP_DATA(cmd[1]);

    IPMI_ADD_RSP_DATA(0); /* Assume success */



    /* If it's too short or it was truncated, return an error. */

    if (cmd_len < 2) {

        rsp[2] = IPMI_CC_REQUEST_DATA_LENGTH_INVALID;

        goto out;

    }

    if (cmd_len > max_cmd_len) {

        rsp[2] = IPMI_CC_REQUEST_DATA_TRUNCATED;

        goto out;

    }



    if ((cmd[0] & 0x03) != 0) {

        /* Only have stuff on LUN 0 */

        rsp[2] = IPMI_CC_COMMAND_INVALID_FOR_LUN;

        goto out;

    }



    netfn = cmd[0] >> 2;



    /* Odd netfns are not valid, make sure the command is registered */

    if ((netfn & 1) || !ibs->netfns[netfn / 2] ||

                        (cmd[1] >= ibs->netfns[netfn / 2]->cmd_nums) ||

                        (!ibs->netfns[netfn / 2]->cmd_handlers[cmd[1]])) {

        rsp[2] = IPMI_CC_INVALID_CMD;

        goto out;

    }



    ibs->netfns[netfn / 2]->cmd_handlers[cmd[1]](ibs, cmd, cmd_len, rsp, rsp_len,

                                                max_rsp_len);



 out:

    k->handle_rsp(s, msg_id, rsp, *rsp_len);



    next_timeout(ibs);

}
