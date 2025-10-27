static void set_bmc_global_enables(IPMIBmcSim *ibs,

                                   uint8_t *cmd, unsigned int cmd_len,

                                   uint8_t *rsp, unsigned int *rsp_len,

                                   unsigned int max_rsp_len)

{

    IPMI_CHECK_CMD_LEN(3);

    set_global_enables(ibs, cmd[2]);

}
