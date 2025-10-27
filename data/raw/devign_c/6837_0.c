static int get_fw_cfg_order(FWCfgState *s, const char *name)

{

    int i;



    if (s->fw_cfg_order_override > 0) {

        return s->fw_cfg_order_override;

    }



    for (i = 0; i < ARRAY_SIZE(fw_cfg_order); i++) {

        if (fw_cfg_order[i].name == NULL) {

            continue;

        }



        if (strcmp(name, fw_cfg_order[i].name) == 0) {

            return fw_cfg_order[i].order;

        }

    }



    /* Stick unknown stuff at the end. */

    error_report("warning: Unknown firmware file in legacy mode: %s", name);

    return FW_CFG_ORDER_OVERRIDE_LAST;

}
